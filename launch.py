#!/usr/bin/env python3
"""InternAgent — master launcher."""

import argparse
import sys
import traceback

from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(
        description='InternAgent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "QA mode:        python launch.py --mode qa --question '...' [--output out.md]\n"
            "Discovery mode: python launch.py --mode discovery --task AutoSeg "
            "--exp_backend claudecode [--mode experiment|report]"
        )
    )
    parser.add_argument(
        '--mode', choices=['qa', 'discovery'], required=True,
        help='Task type: qa for one-shot answers, discovery for idea generation + experiments'
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining  # strip --mode before handing off

    if args.mode == 'qa':
        from launch_qa import main as qa_main
        qa_main()
    elif args.mode == 'discovery':
        from launch_discovery import main as discovery_main
        discovery_main()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        traceback.print_exc()
