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


def atan_series(x: Decimal) -> Decimal:
    """Independent Machin-pi ingredient, rounded by the active Decimal context."""
    power, total, n = x, Decimal(0), 0
    while True:
        updated = total + power / (2 * n + 1)
        if updated == total:
            return total
        total, power, n = updated, -power * x * x, n + 1


def asin_series(x: Decimal) -> Decimal:
    """Positive asin Taylor series; corner inputs have x <= 2**-20.5."""
    term, total, n = x, Decimal(0), 0
    while True:
        updated = total + term
        if updated == total:
            return total
        total = updated
        term *= x * x * (2 * n + 1) ** 2 / (2 * (n + 1) * (2 * n + 3))
        n += 1


def positive_mass(p: Decimal, l: int) -> Decimal:
    q = 1 - p
    return sum(Decimal(math.comb(l, hits)) * p ** hits * q ** (l - hits)
               for hits in range(2, l + 1))


def reference() -> str:
    lines = [BEGIN, 'struct DecimalReference { int k, l; long double log_inclusion; };',
             'const DecimalReference decimal_reference[] = {']
    with localcontext() as context:
        context.prec = 160
        for k in (1, 4, 8, 16, 24, 32, 64, 512, 2048):
            for l in (2, 3, 8, 16, 128, 512):
                p = Decimal(2) ** -k
                total = positive_mass(p, l)
                # The oracle sums every positive mass directly; it never uses
                # the maintained recurrence or complement/log1p expression.
                lines.append(f'    {{{k}, {l}, {total.ln():.60g}L}},')
        lines.extend(['};',
                      'struct DecimalCornerReference { double cosine; int k, l; long double log_inclusion; };',
                      'const DecimalCornerReference decimal_corner_reference[] = {'])
        # This derives acos(c)=2*asin(sqrt((1-c)/2)) using Decimal-only
        # positive Taylor masses and an independent Machin pi. No host libm
        # or maintained angular/logarithmic formula supplies reference values.
        pi = 16 * atan_series(Decimal(1) / 5) - 4 * atan_series(Decimal(1) / 239)
        for gap_bits in (53, 52, 40):
            cosine = 1 - Decimal(2) ** -gap_bits
            theta = 2 * asin_series(((1 - cosine) / 2).sqrt())
            collision = 1 - theta / pi
            for k in (1, 16, 2**31 - 1):
                p = collision ** k
                for l in (2, 512):
                    total = positive_mass(p, l)
                    lines.append(f'    {{{float(cosine).hex()}, {k}, {l}, {total.ln():.60g}L}},')
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
        print('MAGICPIG_DECIMAL_REFERENCE_PASS precision=160 cases=54 direct_double_corners=18')
    else:
        FIXTURE.write_text(source[:start] + generated + source[end:])


if __name__ == '__main__':
    main()
