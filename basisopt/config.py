from pathlib import Path

import toml


class ConfigNamespace:
    """Converts nested dicts to objects with dot-notation access."""

    def __init__(self, data: dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNamespace(value))
            else:
                setattr(self, key, value)

    def as_dict(self) -> dict:
        result = {}
        for key, value in vars(self).items():
            if isinstance(value, ConfigNamespace):
                result[key] = value.as_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"ConfigNamespace({attrs})"

    def __dir__(self):
        return list(vars(self).keys())


class ConfigParser:
    def __init__(self, config_file: str | Path = "config.toml"):
        raw_config = toml.load(config_file)

        for section, values in raw_config.items():
            if isinstance(values, dict):
                setattr(self, section, ConfigNamespace(values))
            else:
                setattr(self, section, values)

    def __dir__(self):
        return list(vars(self).keys())

    def __repr__(self):
        sections = ", ".join(vars(self).keys())
        return f"ConfigParser(sections=[{sections}])"
