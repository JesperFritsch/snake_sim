"""Terminal overwrite demo

Demonstrates:
- single-line overwrite with carriage return
- multi-line overwrite using ANSI cursor movements

Run in PowerShell on Windows. The script uses colorama to enable ANSI on Windows.
"""
import sys
import time

try:
    # colorama is optional but recommended on Windows
    from colorama import init as colorama_init
    colorama_init()
except Exception:
    pass

CSI = "\x1b["  # Control Sequence Introducer


def single_line_demo():
    print("Single-line overwrite demo")
    for i in range(1, 6):
        # \r returns cursor to start of line; end='' prevents newline
        sys.stdout.write(f"Progress: {i}/5\r")
        sys.stdout.flush()
        time.sleep(0.5)
    # finish with a newline so prompt appears correctly
    print("\nDone single-line demo")


def multi_line_demo():
    print("\nMulti-line overwrite demo")
    # print three lines that we'll update
    lines = ["Line A: waiting...", "Line B: waiting...", "Line C: waiting..."]
    for ln in lines:
        print(ln)

    # move cursor up 3 lines, then overwrite each of the 3 lines
    for i in range(5):
        # move cursor up 3 lines: CSI {n}A
        sys.stdout.write(f"{CSI}3A")
        # clear each line and write new content
        for j in range(3):
            # clear line: CSI 2K (erase entire line)
            sys.stdout.write(f"{CSI}2K")
            sys.stdout.write(f"Line {chr(ord('A')+j)}: update {i}\n")
        sys.stdout.flush()
        time.sleep(0.6)

    print("Done multi-line demo")


if __name__ == '__main__':
    single_line_demo()
    multi_line_demo()
