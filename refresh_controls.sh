#!/bin/bash
# Refresh every number that depends on the control and chat runs, in one command, as their seeds land.
#
#   ./refresh_controls.sh            # paper + appendix + PDF; prints the audit summary
#
# The control and chat prose in paper/depth-perception.md is generated (audit_paper_numbers.py
# --write-controls rewrites the <!-- ctl:K --> regions from the logs, including the size of the
# false-discovery-rate family), so nothing here is edited by hand.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY audit_paper_numbers.py --write-controls > /tmp/refresh_audit.log
$PY audit_paper_numbers.py --strict > /tmp/refresh_audit.log || { tail -30 /tmp/refresh_audit.log; exit 1; }
grep -E "claims match" /tmp/refresh_audit.log
$PY make_figures.py > /tmp/refresh_figures.log
$PY make_paper_pdf.py > /tmp/refresh_pdf.log
$PY make_appendix.py > /tmp/refresh_appendix.log
echo "paper, PDF and appendix (docs/index.html) refreshed"
