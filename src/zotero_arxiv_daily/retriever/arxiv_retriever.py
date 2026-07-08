from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from dataclasses import dataclass
from tempfile import TemporaryDirectory
import feedparser
from urllib.request import urlretrieve
from tqdm import tqdm
import os
import re
import time
from loguru import logger


@dataclass
class _RssAuthor:
    name: str


@dataclass
class _RssArxivResult:
    paper_id: str
    title: str
    authors: list[_RssAuthor]
    summary: str
    pdf_url: str
    entry_id: str

    def source_url(self) -> str:
        return f"https://arxiv.org/e-print/{self.paper_id}"


RawArxivPaper = ArxivResult | _RssArxivResult


def _strip_version(paper_id: str) -> str:
    return re.sub(r"v\d+$", "", paper_id)


def _normalize_arxiv_id(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _clean_rss_summary(summary: str) -> str:
    match = re.search(r"Abstract:\s*(.*)", summary, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return summary.strip()


def _rss_entry_to_result(entry) -> _RssArxivResult:
    paper_id = entry.id.removeprefix("oai:arXiv.org:")
    base_id = _strip_version(paper_id)
    authors = []
    if entry.get("authors"):
        authors = [_RssAuthor(a.get("name", "")) for a in entry.authors if a.get("name")]
    elif entry.get("dc_creator"):
        authors = [_RssAuthor(a.strip()) for a in entry.dc_creator.split(",") if a.strip()]

    return _RssArxivResult(
        paper_id=paper_id,
        title=entry.get("title", ""),
        authors=authors,
        summary=_clean_rss_summary(entry.get("summary", "")),
        pdf_url=f"https://arxiv.org/pdf/{base_id}",
        entry_id=entry.get("link", f"https://arxiv.org/abs/{base_id}"),
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_api_batch(
        self,
        client: arxiv.Client,
        paper_ids: list[str],
        max_attempts: int = 3,
    ) -> list[ArxivResult] | None:
        for attempt in range(max_attempts):
            try:
                search = arxiv.Search(id_list=paper_ids)
                return list(client.results(search))
            except Exception as e:
                if attempt == max_attempts - 1:
                    logger.warning(
                        f"Failed to retrieve arXiv metadata for {paper_ids} after "
                        f"{max_attempts} attempts. Fall back to RSS metadata. Error: {e}"
                    )
                    return None
                sleep_seconds = 30 * (2 ** attempt)
                logger.warning(
                    f"Failed to retrieve arXiv metadata for {paper_ids}: {e}. "
                    f"Retrying in {sleep_seconds} seconds."
                )
                time.sleep(sleep_seconds)
        return None

    def _retrieve_raw_papers(self) -> list[RawArxivPaper]:
        client = arxiv.Client(num_retries=2,delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.title:
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        rss_results = [
            _rss_entry_to_result(i)
            for i in feed.entries
            if i.get("arxiv_announce_type","new") == 'new'
        ]
        if self.config.executor.debug:
            rss_results = rss_results[:10]
        rss_results_by_id = {i.paper_id: i for i in rss_results}
        all_paper_ids = [i.paper_id for i in rss_results]

        # Get full information of each paper from arxiv api
        bar = tqdm(total=len(all_paper_ids))
        for i in range(0,len(all_paper_ids),10):
            paper_ids = all_paper_ids[i:i+10]
            batch = self._retrieve_api_batch(client, paper_ids)
            if batch is None:
                raw_papers.extend(rss_results_by_id[paper_id] for paper_id in paper_ids)
            else:
                batch_by_id = {}
                for paper in batch:
                    paper_id = _normalize_arxiv_id(paper.entry_id)
                    batch_by_id[paper_id] = paper
                    batch_by_id[_strip_version(paper_id)] = paper
                for paper_id in paper_ids:
                    raw_papers.append(
                        batch_by_id.get(paper_id)
                        or batch_by_id.get(_strip_version(paper_id))
                        or rss_results_by_id[paper_id]
                    )
            bar.update(len(paper_ids))
            if i + 10 < len(all_paper_ids):
                time.sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper:RawArxivPaper) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_pdf(raw_paper)
        if full_text is None:
            full_text = extract_text_from_tar(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text
        )

def extract_text_from_pdf(paper: RawArxivPaper) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        if paper.pdf_url is None:
            logger.warning(f"No PDF URL available for {paper.title}")
            return None
        try:
            urlretrieve(paper.pdf_url, path)
        except Exception as e:
            logger.warning(f"Failed to download pdf of {paper.title}: {e}")
            return None
        try:
            full_text = extract_markdown_from_pdf(path)
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from pdf: {e}")
            full_text = None
        return full_text

def extract_text_from_tar(paper: RawArxivPaper) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        source_url = paper.source_url()
        if source_url is None:
            logger.warning(f"No source URL available for {paper.title}")
            return None
        try:
            urlretrieve(source_url, path)
        except Exception as e:
            logger.warning(f"Failed to download source of {paper.title}: {e}")
            return None
        try:
            file_contents = extract_tex_code_from_tar(path, paper.entry_id)
            if "all" not in file_contents:
                logger.warning(f"Failed to extract full text of {paper.title} from tar: Main tex file not found.")
                return None
            full_text = file_contents["all"]
        except Exception as e:
            logger.warning(f"Failed to extract full text of {paper.title} from tar: {e}")
            full_text = None
        return full_text
