#!/usr/bin/env python3
"""InternAgent QA mode — direct one-shot question answering."""

import argparse
import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from internagent.mas.agents.dr_agent import DRAgent


def main():
    parser = argparse.ArgumentParser(description='InternAgent QA — one-shot question answering')
    parser.add_argument('--question', '-q', required=True, help='Research question to answer')
    parser.add_argument('--file', '-f', default=None, help='Optional file attachment')
    parser.add_argument('--output', '-o', default=None, help='Write answer to this file path')
    args = parser.parse_args()

    agent = DRAgent(model='o4-mini', config={'mode': 'qa'})
    answer = str(asyncio.run(
        agent.execute({'task': args.question, 'file_path': args.file}, {})
    ))
    print(answer)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(answer)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nQA pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {str(e)}")
        traceback.print_exc()
