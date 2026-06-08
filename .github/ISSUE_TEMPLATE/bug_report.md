---
name: Bug report
about: Report incorrect behaviour, a crash, or a wrong number
title: "[BUG] "
labels: bug
assignees: ''
---

**What happened**
A clear description of the bug.

**Minimal reproducible example**
Ideally offline (use a `DataProvider` with a fixture/CSV so it doesn't depend on live
Yahoo data). Paste the smallest snippet that triggers it:

```python
import quant_reporter as qr
# ...
```

**Expected vs. actual**
- Expected:
- Actual (paste output / traceback):

**Is this a correctness/number issue?**
If a metric or weight looks wrong, say what value you expected and why (formula,
reference, or hand calc). This helps a lot for a quant library.

**Environment**
- quant_reporter version: (`python -c "import quant_reporter as qr; print(qr.__version__)"`)
- Python version:
- OS:

**Anything else**
Logs (enable with `qr.enable_logging()`), screenshots of the HTML report, etc.
