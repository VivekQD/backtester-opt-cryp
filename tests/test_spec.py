import datetime as dt

from btcbt import (Compare, Group, Operand, SessionSpec, CalendarSpec,
                   in_session, to_expr, time_fields)
from btcbt.types import NS


def _ts(y, m, d, hh, mm, tz_offset_h=0):
    t = dt.datetime(y, m, d, hh, mm, tzinfo=dt.timezone.utc)
    return int(t.timestamp()) * NS


def test_to_expr_and_or():
    g_and = Group("AND", [Compare(Operand(const=5), "gt", Operand(const=3)),
                          Compare(Operand(const=2), "le", Operand(const=2))])
    assert to_expr(g_and)(None) is True
    g_and2 = Group("AND", [Compare(Operand(const=5), "lt", Operand(const=3))])
    assert to_expr(g_and2)(None) is False
    g_or = Group("OR", [Compare(Operand(const=1), "gt", Operand(const=9)),
                        Compare(Operand(const=1), "lt", Operand(const=9))])
    assert to_expr(g_or)(None) is True


def test_in_session_days_and_window():
    sess = SessionSpec(start="09:00", squareoff="15:00", days=["MON", "TUE", "WED", "THU", "FRI"])
    # 2026-06-15 is a Monday, 10:00 UTC -> in session
    assert in_session(_ts(2026, 6, 15, 10, 0), sess) is True
    # same day 16:00 -> past window
    assert in_session(_ts(2026, 6, 15, 16, 0), sess) is False
    # 2026-06-13 is Saturday -> out
    assert in_session(_ts(2026, 6, 13, 10, 0), sess) is False


def test_calendar_windows_and_holiday():
    cal = CalendarSpec(tz="UTC", windows=[("09:00", "11:00"), ("13:00", "15:00")],
                       exclude=["2026-06-15"])
    sess = SessionSpec(days=["MON", "TUE", "WED", "THU", "FRI"], calendar=cal)
    # Monday 14:00 normally in 2nd window, but it's a holiday
    assert in_session(_ts(2026, 6, 15, 14, 0), sess) is False
    # Tuesday 14:00 -> in 2nd window
    assert in_session(_ts(2026, 6, 16, 14, 0), sess) is True
    # Tuesday 12:00 -> between windows -> out
    assert in_session(_ts(2026, 6, 16, 12, 0), sess) is False


def test_time_fields():
    tf = time_fields(_ts(2026, 6, 15, 10, 30), "UTC")
    assert tf["hour"] == 10 and tf["minute"] == 30
    assert tf["dow"] == 0          # Monday
    assert tf["month"] == 6
