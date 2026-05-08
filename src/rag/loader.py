"""
src/rag/loader.py
Builds a FAISS index from Investopedia articles + FinDER dataset.

Run once:
    uv run python -m src.rag.loader
"""

import os
import time
import json
import yaml
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
INDEX_PATH      = cfg["rag"]["index_path"]
RAW_CACHE       = cfg["rag"]["raw_cache"]
CHUNK_SIZE      = cfg["rag"]["chunk_size"]
CHUNK_OVERLAP   = cfg["rag"]["chunk_overlap"]
DELAY           = cfg["market"]["delay_seconds"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Investopedia URLs ─────────────────────────────────────────────────────────

ARTICLES = [
    # Finance Q&A
    ("What Is Investing?",              "https://www.investopedia.com/terms/i/investing.asp"),
    ("Dollar-Cost Averaging",           "https://www.investopedia.com/terms/d/dollarcostaveraging.asp"),
    ("Compound Interest",               "https://www.investopedia.com/terms/c/compoundinterest.asp"),
    ("Diversification",                 "https://www.investopedia.com/terms/d/diversification.asp"),
    ("Risk Tolerance",                  "https://www.investopedia.com/terms/r/risktolerance.asp"),
    ("Index Funds",                     "https://www.investopedia.com/terms/i/indexfund.asp"),
    ("ETF vs Mutual Fund",              "https://www.investopedia.com/articles/exchangetradedfunds/08/etf-mutual-fund-difference.asp"),
    ("Stocks vs Bonds",                 "https://www.investopedia.com/ask/answers/09/difference-between-bond-stock-market.asp"),
    ("Liquidity",                       "https://www.investopedia.com/terms/l/liquidity.asp"),
    ("Bull vs Bear Market",             "https://www.investopedia.com/insights/digging-deeper-bull-and-bear-markets/"),
    ("Time Value of Money",             "https://www.investopedia.com/terms/t/timevalueofmoney.asp"),
    ("Inflation",                       "https://www.investopedia.com/terms/i/inflation.asp"),
    # Portfolio
    ("Asset Allocation",                "https://www.investopedia.com/terms/a/assetallocation.asp"),
    ("Portfolio Rebalancing",           "https://www.investopedia.com/terms/r/rebalancing.asp"),
    ("Modern Portfolio Theory",         "https://www.investopedia.com/terms/m/modernportfoliotheory.asp"),
    ("Beta",                            "https://www.investopedia.com/terms/b/beta.asp"),
    ("Alpha",                           "https://www.investopedia.com/terms/a/alpha.asp"),
    ("Sharpe Ratio",                    "https://www.investopedia.com/terms/s/sharperatio.asp"),
    ("Standard Deviation",              "https://www.investopedia.com/terms/s/standarddeviation.asp"),
    ("Correlation",                     "https://www.investopedia.com/terms/c/correlation.asp"),
    ("How to Build a Portfolio",        "https://www.investopedia.com/articles/basics/06/invest1000.asp"),
    ("Expense Ratio",                   "https://www.investopedia.com/terms/e/expenseratio.asp"),
    # Market
    ("P/E Ratio",                       "https://www.investopedia.com/terms/p/price-earningsratio.asp"),
    ("Market Capitalization",           "https://www.investopedia.com/terms/m/marketcapitalization.asp"),
    ("EPS",                             "https://www.investopedia.com/terms/e/eps.asp"),
    ("Dividend Yield",                  "https://www.investopedia.com/terms/d/dividendyield.asp"),
    ("P/B Ratio",                       "https://www.investopedia.com/terms/p/price-to-bookratio.asp"),
    ("Moving Averages",                 "https://www.investopedia.com/terms/m/movingaverage.asp"),
    ("Support and Resistance",          "https://www.investopedia.com/trading/support-and-resistance-basics/"),
    ("Volume in Stock Trading",         "https://www.investopedia.com/terms/v/volume.asp"),
    ("Market Sentiment",                "https://www.investopedia.com/terms/m/marketsentiment.asp"),
    # Goal Planning
    ("Financial Goals",                 "https://www.investopedia.com/terms/f/financial_plan.asp"),
    ("Emergency Fund",                  "https://www.investopedia.com/terms/e/emergency_fund.asp"),
    ("Retirement Planning",             "https://www.investopedia.com/terms/r/retirement-planning.asp"),
    ("Rule of 72",                      "https://www.investopedia.com/terms/r/ruleof72.asp"),
    ("Net Worth",                       "https://www.investopedia.com/terms/n/networth.asp"),
    ("Saving vs Investing",             "https://www.investopedia.com/articles/investing/022516/saving-vs-investing-understanding-key-differences.asp"),
    ("Risk vs Reward",                  "https://www.investopedia.com/terms/r/riskreturntradeoff.asp"),
    ("SMART Financial Goals",           "https://www.investopedia.com/articles/personal-finance/100516/setting-financial-goals/"),
    # News
    ("How Interest Rates Affect Markets","https://www.investopedia.com/articles/stocks/09/how-interest-rates-affect-markets.asp"),
    ("How the Stock Market Works",      "https://www.investopedia.com/articles/investing/082614/how-stock-market-works.asp"),
    ("Federal Reserve",                 "https://www.investopedia.com/terms/f/federalreservebank.asp"),
    ("GDP",                             "https://www.investopedia.com/ask/answers/what-is-gdp-why-its-important-to-economists-investors/"),
    # Tax
    ("Capital Gains Tax",               "https://www.investopedia.com/terms/c/capital_gains_tax.asp"),
    ("401(k)",                          "https://www.investopedia.com/terms/1/401kplan.asp"),
    ("IRA",                             "https://www.investopedia.com/terms/i/ira.asp"),
    ("Roth IRA",                        "https://www.investopedia.com/terms/r/rothira.asp"),
    ("Tax-Loss Harvesting",             "https://www.investopedia.com/terms/t/taxgainlossharvesting.asp"),
    ("Roth vs Traditional IRA",         "https://www.investopedia.com/retirement/roth-vs-traditional-ira-which-is-right-for-you/"),
    ("52-Week High and Low",            "https://www.investopedia.com/terms/1/52weekhighlow.asp"),
# Derivatives
    ("Swaps",  "https://www.investopedia.com/terms/s/swap.asp"),
    ("Options",  "https://www.investopedia.com/terms/o/option.asp"),
    ("Futures",  "https://www.investopedia.com/terms/f/futures.asp"),

# Personal Finance
    ("How to Save Money",  "https://www.investopedia.com/articles/personal-finance/100516/setting-financial-goals/"),
    ("Budgeting Basics",  "https://www.investopedia.com/terms/b/budget.asp"),
    ("50/30/20 Rule",  "https://www.investopedia.com/ask/answers/022916/what-502030-budget-rule.asp"),
    ("Credit Score",  "https://www.investopedia.com/terms/c/credit_score.asp"),
    ("Debt Management",  "https://www.investopedia.com/terms/d/debtmanagement.asp"),
    ("Credit Cards",  "https://www.investopedia.com/terms/c/creditcard.asp"),
    ("Repurchase Agreement (Repo)",  "https://www.investopedia.com/terms/r/repurchaseagreement.asp"),

    ("Real Estate Investing", "https://www.investopedia.com/terms/r/realestate.asp"),
    ("REITs", "https://www.investopedia.com/terms/r/reit.asp"),
    ("How to Buy a Home", "https://www.investopedia.com/articles/mortgages-real-estate/08/first-time-homebuyer-tips.asp"),
    ("Mortgage Basics", "https://www.investopedia.com/terms/m/mortgage.asp"),
    ("Art as an Investment", "https://www.investopedia.com/articles/pf/08/fine-art.asp"),
    ("Commodities", "https://www.investopedia.com/terms/c/commodity.asp"),
    ("Gold as Investment", "https://www.investopedia.com/articles/basics/09/precious-metals-gold-silver-platinum.asp"),
    ("Cryptocurrency Basics", "https://www.investopedia.com/terms/c/cryptocurrency.asp"),

    ("Retirement Planning", "https://www.investopedia.com/terms/r/retirement-planning.asp"),
    ("Social Security", "https://www.investopedia.com/terms/s/socialsecurity.asp"),
    ("Medicare Basics", "https://www.investopedia.com/terms/m/medicare.asp"),
    ("Required Minimum Distributions", "https://www.investopedia.com/terms/r/requiredminimumdistribution.asp"),
    ("Annuities", "https://www.investopedia.com/terms/a/annuity.asp"),
    ("When to Retire", "https://www.investopedia.com/articles/retirement/when-can-you-retire.asp"),

    ("Life Insurance", "https://www.investopedia.com/terms/l/lifeinsurance.asp"),
    ("Health Insurance Basics", "https://www.investopedia.com/terms/h/healthinsurance.asp"),
    ("Disability Insurance", "https://www.investopedia.com/terms/d/disability-insurance.asp"),

    ("Student Loans", "https://www.investopedia.com/terms/s/student-debt.asp"),
    ("How to Get Out of Debt", "https://www.investopedia.com/articles/pf/how-to-get-out-of-debt.asp"),
    ("Good Debt vs Bad Debt", "https://www.investopedia.com/articles/pf/12/good-debt-bad-debt.asp"),

    ("Wills and Trusts", "https://www.investopedia.com/terms/w/will.asp"),
    ("Estate Planning Basics", "https://www.investopedia.com/terms/e/estateplanning.asp"),
]

# ── Investopedia Scraper ──────────────────────────────────────────────────────

def scrape_article(title: str, url: str) -> dict | None:
    try:
        print(f"  [{title}]")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside", "form", "header"]):
        tag.decompose()

    body = (
        soup.find("article")
        or soup.find(class_="article-body-content")
        or soup.find(id="article-body")
        or soup.find("main")
    )

    if not body:
        print(f"  WARNING: no body found")
        return None

    text = body.get_text(separator="\n", strip=True)
    if len(text) < 300:
        print(f"  WARNING: too short ({len(text)} chars)")
        return None

    return {"title": title, "url": url, "text": text, "source": "investopedia"}


def load_investopedia() -> list[dict]:
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(RAW_CACHE):
        print(f"Loading cached articles from {RAW_CACHE}")
        with open(RAW_CACHE) as f:
            return json.load(f)

    print(f"Scraping {len(ARTICLES)} Investopedia articles...")
    articles = []
    for i, (title, url) in enumerate(ARTICLES, 1):
        print(f"[{i}/{len(ARTICLES)}]", end=" ")
        article = scrape_article(title, url)
        if article:
            articles.append(article)
        time.sleep(DELAY)

    with open(RAW_CACHE, "w") as f:
        json.dump(articles, f, indent=2)
    print(f"Saved {len(articles)} articles to {RAW_CACHE}")
    return articles


# ── Investopedia Chunker ──────────────────────────────────────────────────────

def chunk_investopedia(articles: list[dict]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )
    docs = []
    for article in articles:
        for i, chunk in enumerate(splitter.split_text(article["text"])):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title":  article["title"],
                    "url":    article["url"],
                    "source": "investopedia",
                    "chunk":  i,
                }
            ))
    print(f"Investopedia: {len(docs)} chunks from {len(articles)} articles")
    return docs


# ── FinDER Loader ─────────────────────────────────────────────────────────────

def load_finder() -> list[Document]:
    print("Loading FinDER dataset from HuggingFace...")
    ds = load_dataset("Linq-AI-Research/FinDER", split="train")
    docs = []
    for row in ds:
        for ref in row["references"]:
            if not ref.strip():
                continue
            docs.append(Document(
                page_content=ref.strip(),
                metadata={
                    "title":    row["_id"],
                    "source":   "finder",
                    "category": row["category"],
                    "type":     row["type"],
                    "answer":   row["answer"],
                }
            ))
    print(f"FinDER: {len(docs)} passages loaded")
    return docs


# ── Build Index ───────────────────────────────────────────────────────────────

def build_index(docs: list[Document], force: bool = False) -> None:
    index_file = os.path.join(INDEX_PATH, "index.faiss")
    if not force and os.path.exists(index_file):
        print(f"Index already exists at {INDEX_PATH}/, skipping build. Pass force=True to rebuild.")
        return

    print(f"\nEmbedding {len(docs)} total chunks...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    vectorstore = FAISS.from_documents(docs, embeddings)
    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)
    print(f"Index saved to {INDEX_PATH}/")


# ── Smoke Test ────────────────────────────────────────────────────────────────

def smoke_test() -> None:
    print("\nSmoke test...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    vectorstore = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    queries = [
        "What is dollar cost averaging?",
        "How does a Roth IRA work?",
        "Analyze CrowdStrike revenue growth",
    ]
    for q in queries:
        results = retriever.invoke(q)
        top = results[0]
        print(f"\n  Q: {q}")
        print(f"  -> [{top.metadata['source']}] {top.page_content[:100].replace(chr(10),' ')}...")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Finnie RAG Loader")
    print("=" * 55)

    # 1. Investopedia
    articles = load_investopedia()
    investopedia_docs = chunk_investopedia(articles)

    # 2. FinDER
    finder_docs = load_finder()

    # 3. Merge + build
    all_docs = investopedia_docs + finder_docs
    print(f"\nTotal documents: {len(all_docs)}")
    build_index(all_docs)

    # 4. Smoke test
    smoke_test()

    print("\n" + "=" * 55)
    print("  Done! RAG index ready.")
    print("=" * 55)