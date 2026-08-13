# exceptions


class MethodNotAvailable(Exception):
    def __init__(self, estr):
        super().__init__(f"Method not available: {estr}")
        self.method_str = estr


class PropertyNotAvailable(Exception):
    def __init__(self, pstr):
        super().__init__(f"Property not available: {pstr}")
        self.property_str = pstr


class EmptyCalculation(Exception):
    pass


class FailedCalculation(Exception):
    pass


class BackendNotFound(Exception):
    """A requested calculation backend could not be imported/initialised."""

    pass


class ElementNotSet(Exception):
    pass


class EmptyBasis(Exception):
    pass


class InvalidResult(Exception):
    pass


class InvalidMethodString(Exception):
    pass


class DataNotFound(Exception):
    pass


class InvalidDiatomic(Exception):
    pass
