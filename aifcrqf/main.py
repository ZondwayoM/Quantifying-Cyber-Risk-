"""
AIFCRQF — Entry point.

Simple usage (positional shorthand):
    python main.py all weak          # all domains, weak controls  (M=0.30)
    python main.py all medium        # all domains, medium controls (M=0.60)
    python main.py all strong        # all domains, strong controls (M=0.80)

    python main.py fraud strong      # single domain, specific maturity
    python main.py credit medium
    python main.py aml weak
    python main.py trading strong

Full flag usage (for custom paths or iterations):
    python main.py --domain fraud --iso-maturity strong --iterations 50000
    python main.py --data path/to/data.csv --iso-maturity medium

Launch dashboard (after pipeline run):
    streamlit run run_streamlit_dashboard.py
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config.settings import LOGS_DIR, MONTE_CARLO_ITERATIONS
from pipeline.orchestrator import Orchestrator

# Console + file logging, UTF-8 safe
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1, closefd=False)
        ),
        logging.FileHandler(LOGS_DIR / "aifcrqf.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

DOMAIN_DATA = {
    "fraud":   "data/raw/creditcard.csv",
    "credit":  "data/raw/german_credit.csv",
    "aml":     "data/raw/aml_ibm.csv",
    "trading": "data/processed/trading_features.csv",
}

_MATURITY_ALIASES = {"weak", "medium", "strong"}
_DOMAIN_ALIASES   = set(DOMAIN_DATA.keys()) | {"all"}


def _resolve_positional(args_list: list[str]) -> tuple[str | None, str | None]:
    """Extract optional positional shorthand tokens (e.g. 'all strong') before argparse runs."""
    tokens = [t.lower() for t in args_list if not t.startswith("-")]
    domain   = next((t for t in tokens if t in _DOMAIN_ALIASES),   None)
    maturity = next((t for t in tokens if t in _MATURITY_ALIASES), None)
    if domain or maturity:
        return domain, maturity
    return None, None


def parse_args() -> argparse.Namespace:
    pos_domain, pos_maturity = _resolve_positional(sys.argv[1:])

    # Strip positional tokens so argparse only sees flags
    cleaned = [
        a for a in sys.argv[1:]
        if a.lower() not in (_DOMAIN_ALIASES | _MATURITY_ALIASES)
    ]

    parser = argparse.ArgumentParser(
        description="AIFCRQF — run with positional args or flags (see module docstring)"
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=list(DOMAIN_DATA.keys()) + ["all"],
        default=None,
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to input CSV (overrides --domain).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/",
    )
    parser.add_argument(
        "--iso-maturity",
        type=str,
        default="medium",
        choices=["weak", "medium", "strong"],
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=MONTE_CARLO_ITERATIONS,
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the pipeline finishes.",
    )

    ns = parser.parse_args(cleaned)

    if pos_domain and ns.domain is None:
        ns.domain = pos_domain
    if pos_maturity and ns.iso_maturity == "medium":   # only override the default
        ns.iso_maturity = pos_maturity

    return ns


def run_single(data_path: str, args: argparse.Namespace) -> None:
    path = Path(data_path)
    if not path.exists():
        logger.error("Data file not found: %s", path)
        sys.exit(1)
    logger.info("=== AIFCRQF Pipeline Starting ===")
    logger.info("Data     : %s", path)
    logger.info("Target   : %s", args.target or "auto-detect")
    logger.info("Output   : %s", args.output)
    logger.info("Maturity : %s", args.iso_maturity)
    logger.info("MC iters : %d", args.iterations)
    Orchestrator(
        data_path=str(path),
        target_col=args.target,
        output_dir=args.output,
        iso_maturity=args.iso_maturity,
        mc_iterations=args.iterations,
    ).run()
    logger.info("=== AIFCRQF Pipeline Complete ===")


def launch_dashboard() -> None:
    """Launch Streamlit dashboard in a new process (non-blocking)."""
    logger.info("Launching Streamlit dashboard → http://localhost:8501")
    subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "run_streamlit_dashboard.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    args = parse_args()

    if args.domain == "all":
        for domain, data in DOMAIN_DATA.items():
            logger.info(">>> Running domain: %s", domain.upper())
            try:
                run_single(data, args)
            except Exception as exc:
                logger.error("Domain %s failed — skipping: %s", domain.upper(), exc)
    elif args.domain:
        run_single(DOMAIN_DATA[args.domain], args)
    elif args.data:
        run_single(args.data, args)
    else:
        logger.error(
            "No domain specified. Examples:\n"
            "  python main.py all strong\n"
            "  python main.py fraud medium\n"
            "  python main.py --domain credit --iso-maturity weak"
        )
        sys.exit(1)

    launch_dashboard()


if __name__ == "__main__":
    main()
