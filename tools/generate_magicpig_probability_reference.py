#!/usr/bin/env python3
"""Reproduce the independent 160-digit positive-binomial reference literals."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import math
from pathlib import Path

BEGIN = '// BEGIN DECIMAL PROBABILITY REFERENCE'
END = '// END DECIMAL PROBABILITY REFERENCE'
FIXTURE = Path(__file__).resolve().parents[1] / 'tests/traceability/test_magicpig_probability_cpu.cpp'


def reference() -> str:
    lines = [BEGIN, 'struct DecimalReference { int k, l; long double log_inclusion; };',
             'const DecimalReference decimal_reference[] = {']
    with localcontext() as context:
        context.prec = 160
        for k in (1, 4, 8, 16, 24, 32, 64, 512, 2048):
            for l in (2, 3, 8, 16, 128, 512):
                p = Decimal(2) ** -k
                q = 1 - p
                total = sum(Decimal(math.comb(l, hits)) * p ** hits * q ** (l - hits)
                            for hits in range(2, l + 1))
                # The oracle sums every positive mass directly; it never uses
                # the maintained recurrence or complement/log1p expression.
                lines.append(f'    {{{k}, {l}, {total.ln():.60g}L}},')
    return '\n'.join([*lines, '};', END])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = FIXTURE.read_text()
    start = source.index(BEGIN)
    end = source.index(END, start) + len(END)
    generated = reference()
    if args.check:
        if source[start:end] != generated:
            raise SystemExit('probability Decimal reference literals differ')
        print('MAGICPIG_DECIMAL_REFERENCE_PASS precision=160 cases=54')
    else:
        FIXTURE.write_text(source[:start] + generated + source[end:])


if __name__ == '__main__':
    main()
