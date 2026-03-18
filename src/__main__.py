"""CLI for lipread-ai."""
import sys, json, argparse
from .core import LipreadAi

def main():
    parser = argparse.ArgumentParser(description="LipRead AI — Video Lip Reading. AI lip reading from video for accessibility and transcription.")
    parser.add_argument("command", nargs="?", default="status", choices=["status", "run", "info"])
    parser.add_argument("--input", "-i", default="")
    args = parser.parse_args()
    instance = LipreadAi()
    if args.command == "status":
        print(json.dumps(instance.get_stats(), indent=2))
    elif args.command == "run":
        print(json.dumps(instance.process(input=args.input or "test"), indent=2, default=str))
    elif args.command == "info":
        print(f"lipread-ai v0.1.0 — LipRead AI — Video Lip Reading. AI lip reading from video for accessibility and transcription.")

if __name__ == "__main__":
    main()
