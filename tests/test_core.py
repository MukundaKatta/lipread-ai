"""Tests for LipreadAi."""
from src.core import LipreadAi
def test_init(): assert LipreadAi().get_stats()["ops"] == 0
def test_op(): c = LipreadAi(); c.process(x=1); assert c.get_stats()["ops"] == 1
def test_multi(): c = LipreadAi(); [c.process() for _ in range(5)]; assert c.get_stats()["ops"] == 5
def test_reset(): c = LipreadAi(); c.process(); c.reset(); assert c.get_stats()["ops"] == 0
def test_service_name(): c = LipreadAi(); r = c.process(); assert r["service"] == "lipread-ai"
