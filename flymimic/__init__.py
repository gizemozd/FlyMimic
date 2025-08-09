from dm_control import suite

from flymimic.tasks import fly

suite._DOMAINS["fly"] = fly
