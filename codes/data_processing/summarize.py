import os
import pandas as pd
import asyncio
from typing import List
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException, Timeout
from tqdm.asyncio import tqdm_asyncio
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pathlib import Path

# ===================== Core Configuration =====================
# Async configuration
ASYNC_TIMEOUT = 300  # Timeout for single async task (seconds)
RETRY_ATTEMPTS = 3  # Retry times for failed requests
RETRY_WAIT = (1, 5)  # Exponential backoff for retries (1-5 seconds)
semaphore = asyncio.Semaphore(5)  # Async rate limit (max 5 concurrent requests)

# Data path configuration
DATASET_NAME = "哪吒之魔童闹海的影评"
DATA_ROOT_PATH = str(Path(__file__).parent.parent.parent.absolute())
# print(DATA_ROOT_PATH.absolute())
RAW_ORIGINAL_DATA_PATH = os.path.join(DATA_ROOT_PATH, f"data/reviews_after_annotation/{DATASET_NAME}_attributes_for_each_review2.xlsx")
# print(RAW_ORIGINAL_DATA_PATH)
# Generate summary file path (original filename + _summary.xlsx)

SUMMARY_SAVE_PATH = os.path.join(DATA_ROOT_PATH, f"data/reviews_after_summary(must_before_split)/{DATASET_NAME}_attributes_for_each_review2.xlsx")

# DeepSeek configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"  # Support async invocation


# ===================== Utility Functions =====================
def load_aspect_info() -> str:
    """Load aspect definitions from excel file for summary prompt"""
    aspects_file = os.path.join(DATA_ROOT_PATH, f"aspects_{DATASET_NAME}.xlsx")
    try:
        df_aspects = pd.read_excel(aspects_file, header=0)
        aspects_list = []
        for idx, row in df_aspects.iterrows():
            aspect_name = row["aspects"]
            aspect_desc = row["description"]
            aspects_list.append(f"{idx + 1}. {aspect_name}：{aspect_desc}")
        aspects_full_text = "\n".join(aspects_list)
        print(f"✅ Successfully loaded {len(aspects_list)} aspect definitions")
        return aspects_full_text
    except Exception as e:
        print(f"❌ Failed to load aspects file: {str(e)}")
        raise


def get_summary_prompt_template(aspects_full_text: str) -> PromptTemplate:
    """Create prompt template for text summarization with aspect constraints"""
    prompt_template = PromptTemplate.from_template("""
    任务：为以下影评生成简洁摘要，核心要求是**完整保留与指定9个方面相关的所有信息**，便于后续进行方面级情感分析。

    关键说明：
    1. 必须保留的9个方面及定义如下：
    {aspects_full_text}

    2. 思维链要求（请按以下步骤思考后再生成摘要）：
    - 第一步：分析原影评中提到了上述哪些方面（标注序号和名称）；
    - 第二步：提取每个被提及方面的核心观点（包括情感倾向相关的关键词，如“特效惊艳”“剧情混乱”）；
    - 第三步：浓缩为简洁摘要，不遗漏任何被提及方面的核心信息，不添加无关内容。

    3. 输出要求：
    - 摘要长度控制在200字以内（确保适配BERT的512token限制）；
    - 仅输出摘要文本，不包含思维链分析过程、序号、格式标记等额外内容；
    - 语言流畅，忠于原文意思，不改变情感倾向。

    待摘要影评：{review}
    """)
    return prompt_template


# ===================== Async Core Functions =====================
async def init_deepseek_llm_async() -> ChatOpenAI:
    """Initialize DeepSeek LLM asynchronously with chain-of-thought enabled"""
    if not DEEPSEEK_API_KEY:
        raise ValueError("❌ Please set environment variable DEEPSEEK_API_KEY first")
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        timeout=ASYNC_TIMEOUT,
        extra_body={"enable_thinking": True}
    )
    print(f"✅ Successfully initialized {MODEL_NAME} (async version + chain-of-thought)")
    return llm


async def invoke_summary_safe_with_retry(
        llm: ChatOpenAI,
        prompt_template: PromptTemplate,
        review: str,
        review_idx: int,
        aspects_full_text: str
) -> str:
    """Invoke LLM to generate summary with rate limit, retry and timeout handling"""
    async with semaphore:
        async for attempt in AsyncRetrying(
                stop=stop_after_attempt(RETRY_ATTEMPTS),
                wait=wait_exponential(multiplier=1, min=RETRY_WAIT[0], max=RETRY_WAIT[1]),
                retry=retry_if_exception_type((RequestException, Timeout, Exception)),
                reraise=False
        ):
            with attempt:
                try:
                    # Render prompt with review and aspect info
                    prompt_str = prompt_template.format(
                        aspects_full_text=aspects_full_text,
                        review=review
                    )
                    # Async LLM invocation with timeout
                    resp = await asyncio.wait_for(
                        llm.ainvoke(prompt_str),
                        timeout=ASYNC_TIMEOUT
                    )
                    summary = resp.content.strip()
                    # Fallback: use first 200 chars of original text if summary is empty
                    if not summary:
                        summary = review[:200].strip()
                    return summary
                except asyncio.TimeoutError:
                    print(f"\n⚠️ Timeout warning: Summary for review #{review_idx + 1} exceeded {ASYNC_TIMEOUT}s | Fragment: {review[:30]}...")
                    return review[:200].strip()
                except Exception as e:
                    if attempt.retry_state.attempt_number >= RETRY_ATTEMPTS:
                        print(
                            f"\n❌ Invocation failed: Review #{review_idx + 1} failed after {RETRY_ATTEMPTS} retries | Error: {str(e)[:100]}")
                        return review[:200].strip()
                    raise


# ===================== Async Batch Generate Summary File =====================
async def generate_text_summary_file_async():
    """Generate summary file with original data structure (keep all columns except replace text with summary)"""
    # Load original data with all columns
    try:
        df_original = pd.read_excel(RAW_ORIGINAL_DATA_PATH, header=0)
        print(f"✅ Successfully loaded original data with {len(df_original)} records (all columns retained)")
        text_col = df_original.columns[0]  # Assume text column is the first column
        raw_texts = df_original[text_col].astype(str).tolist()
    except Exception as e:
        print(f"❌ Failed to load original data: {str(e)}")
        return

    # Initialize LLM, prompt template and aspect info
    llm = await init_deepseek_llm_async()
    aspects_full_text = load_aspect_info()
    prompt_template = get_summary_prompt_template(aspects_full_text)

    # Create async tasks for batch summarization
    tasks = [
        invoke_summary_safe_with_retry(
            llm=llm,
            prompt_template=prompt_template,
            review=review,
            review_idx=idx,
            aspects_full_text=aspects_full_text
        ) for idx, review in enumerate(raw_texts)
    ]

    # Execute async tasks with progress bar
    print(f"\n🚀 Start async summary generation (total {len(tasks)} reviews, {semaphore._value} concurrent requests)")
    summaries = await tqdm_asyncio.gather(
        *tasks,
        desc="Processing review summaries asynchronously",
        total=len(tasks),
        ncols=100,
        colour="green"
    )

    # Create summary dataframe with original structure
    df_summary = df_original.copy()
    df_summary[text_col] = summaries
    # Clean text format (remove extra spaces and newlines)
    df_summary[text_col] = df_summary[text_col].str.strip().str.replace("\n", " ").str.replace("  ", " ")

    # Save summary file as xlsx
    df_summary.to_excel(SUMMARY_SAVE_PATH, index=False, engine="openpyxl")
    print(f"\n✅ Summary file saved to: {SUMMARY_SAVE_PATH}")

    # Output statistics
    success_num = sum(
        [1 for idx, s in enumerate(summaries) if len(s) < 200 or (len(s) == 200 and s != raw_texts[idx][:200])]
    )
    fail_num = len(summaries) - success_num
    print(f"\n📊 Statistics:")
    print(f"  - Total records: {len(summaries)}")
    print(f"  - Successfully generated summaries: {success_num}")
    print(f"  - Fallback processed (timeout/failure): {fail_num}")


# ===================== Execution Entry =====================
if __name__ == "__main__":
    # Fix async event loop issue for Windows platform (Python 3.8+)
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Run async main function
    asyncio.run(generate_text_summary_file_async())