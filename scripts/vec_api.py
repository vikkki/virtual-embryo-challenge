#!/usr/bin/env python
"""Virtual Embryo API helper.

  vec_api.py phase | boards | me | quota | manifest | submissions
  vec_api.py download [KEY ...]          # no keys = everything in the manifest
  vec_api.py check FILE TASK [--setting heart|embryo] [--mode interp|extrap] [--split val|test]
  vec_api.py status SUBMISSION_ID

Official uploads are deliberately NOT exposed here; use vec.api.submit() by hand.
"""
import argparse
import json

from vec import api


def show(x):
    print(json.dumps(x, indent=2, default=str))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("cmd")
    p.add_argument("args", nargs="*")
    p.add_argument("--setting")
    p.add_argument("--mode")
    p.add_argument("--split")
    a = p.parse_args()
    if a.cmd in {"phase", "boards", "me", "quota", "manifest", "submissions"}:
        show(getattr(api, a.cmd)())
    elif a.cmd == "download":
        keys = a.args
        if not keys:
            m = api.manifest()
            files = m.get("files", m) if isinstance(m, dict) else m
            keys = [f["key"] if isinstance(f, dict) else f for f in files]
        for k in keys:
            print("->", api.download(k))
    elif a.cmd == "check":
        f, task = a.args
        show(api.format_check(f, task, a.setting, a.mode, a.split))
    elif a.cmd == "status":
        show(api.submission(a.args[0]))
    else:
        p.error(f"unknown command {a.cmd}")


if __name__ == "__main__":
    main()
