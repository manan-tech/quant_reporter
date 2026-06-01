# Security Policy

## Supported versions

`quant_reporter` is maintained by a single author on a best-effort basis. Security
fixes are applied to the latest released `2.x` line only.

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| < 2.1   | :x:                |

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue
for anything exploitable.

- Email: **manan@targetpeak.in** with the subject line `quant_reporter security`.
- Include: affected version, a description, and a minimal reproduction if possible.

You can expect an initial acknowledgement within **7 days**. As a single-maintainer
project, fix timelines are best-effort and depend on severity; please allow
reasonable time for a coordinated fix before any public disclosure.

## Scope

`quant_reporter` fetches market data from third-party sources (Yahoo Finance via
`yfinance` by default) and renders local HTML reports. Relevant concerns include:

- Treat all fetched market data and any HTML reports built from untrusted inputs
  (e.g. attacker-controlled ticker strings or display names) as untrusted; reports
  are intended to be opened by their author, not served to third parties unsanitised.
- This library performs **no** authentication, stores no credentials, and is **not**
  financial advice (see the README Disclaimer). Use at your own risk.
