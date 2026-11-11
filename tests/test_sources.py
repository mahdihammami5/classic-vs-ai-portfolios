import io
import zipfile

import pytest

from pfopt.data.sources import parse_fama_french_rf, to_yahoo
from pfopt.data.universe import Membership


def test_parse_fama_french_rf_reads_daily_rows_only():
    text = (
        "This file was created by ...\n\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "20200102,    0.86,   -0.97,   -0.35,    0.01\n"
        "20200103,   -0.67,    0.30,    0.00,    0.02\n\n"
        "Copyright 2026\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("F-F.csv", text)
    rf = parse_fama_french_rf(buf.getvalue())
    assert len(rf) == 2
    assert rf.iloc[1] == pytest.approx(0.0002)


def test_yahoo_ticker_mapping():
    assert to_yahoo("BRK.B") == "BRK-B"
    assert to_yahoo("AAPL") == "AAPL"
    assert to_yahoo("FB") == "META"  # renamed company keeps its history under the new symbol


def test_membership_is_point_in_time():
    m = Membership.from_dict({"2010-01-04": ["A", "B"], "2015-06-01": ["A", "C"]})
    assert m.members_at("2009-12-31") == frozenset()
    assert m.members_at("2012-01-01") == {"A", "B"}
    assert m.members_at("2015-06-01") == {"A", "C"}
    assert m.ever_members("2011-01-01", "2016-01-01") == {"A", "B", "C"}
