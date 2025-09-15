"""
API Contract Test for CyborgDB Python SDK

This test verifies the complete public API surface of the CyborgDB Python SDK.
It checks all public functions, their signatures, types, and parameter requirements.
Any breaking changes to the API will cause this test to fail.

The test structure follows the documentation at:
../cyborgdb-docs/versions/v0.12.x/service/python-sdk
"""

import inspect
from typing import get_type_hints, Union, List, Dict, Any


class APIContractViolation:
    """Represents a single API contract violation."""

    def __init__(self, component: str, issue: str):
        self.component = component
        self.issue = issue

    def __str__(self):
        return f"[{self.component}] {self.issue}"


class APIContractTester:
    """Main test class that checks the entire API contract."""

    def __init__(self):
        self.violations = []

    def add_violation(self, component: str, issue: str):
        """Add a violation to the list."""
        self.violations.append(APIContractViolation(component, issue))

    def check_class_exists(self, module_name: str, class_name: str) -> type:
        """Check if a class exists and is accessible."""
        try:
            module = __import__(module_name, fromlist=[class_name])
            if not hasattr(module, class_name):
                self.add_violation(
                    f"{module_name}.{class_name}", "Class does not exist"
                )
                return None
            return getattr(module, class_name)
        except ImportError as e:
            self.add_violation(f"{module_name}.{class_name}", f"Import error: {e}")
            return None

    def check_function_signature(
        self,
        obj: Any,
        method_name: str,
        expected_params: Dict[str, Any],
        expected_return: Any = None,
        is_property: bool = False,
        is_static: bool = False,
    ):
        """Check if a function/method has the expected signature.

        expected_params format:
        {
            'param_name': {
                'type': type or typing hint,
                'optional': bool,
                'default': expected default value (if optional),
            }
        }
        """
        if obj is None:
            return

        component = f"{obj.__name__ if hasattr(obj, '__name__') else obj.__class__.__name__}.{method_name}"

        # Check if method exists
        if not hasattr(obj, method_name):
            self.add_violation(component, "Method/property does not exist")
            return

        attr = getattr(obj, method_name)

        # Check if it's a property when expected
        if is_property:
            if not isinstance(inspect.getattr_static(obj, method_name), property):
                self.add_violation(component, "Expected property but got method")
                return
            # Properties don't have parameters to check, but could check return type
            if expected_return:
                try:
                    # For properties, we need to check the fget function
                    prop = inspect.getattr_static(obj, method_name)
                    if hasattr(prop, "fget") and prop.fget:
                        type_hints = get_type_hints(prop.fget)
                        if "return" in type_hints:
                            actual_return = type_hints["return"]
                            if not self._types_compatible(
                                actual_return, expected_return
                            ):
                                self.add_violation(
                                    component,
                                    f"Property return type mismatch: expected {expected_return}, got {actual_return}",
                                )
                except Exception:
                    pass  # Type hints might not be available
            return

        # Check if it's static when expected
        if is_static:
            if not isinstance(inspect.getattr_static(obj, method_name), staticmethod):
                self.add_violation(
                    component, "Expected static method but got instance method"
                )

        # Get the actual function (unwrap if needed)
        if hasattr(attr, "__func__"):
            func = attr.__func__
        else:
            func = attr

        # Check parameters
        try:
            sig = inspect.signature(func)
            params = dict(sig.parameters)

            # Remove 'self' from instance methods
            if "self" in params and not is_static:
                del params["self"]

            # Check that parameter names and order match exactly
            expected_param_names = list(expected_params.keys())
            actual_param_names = list(params.keys())

            # Check parameter order
            for i, expected_name in enumerate(expected_param_names):
                if i >= len(actual_param_names):
                    self.add_violation(component, f"Missing parameter: {expected_name}")
                    continue
                if actual_param_names[i] != expected_name:
                    self.add_violation(
                        component,
                        f"Parameter order mismatch at position {i}: "
                        f"expected '{expected_name}', got '{actual_param_names[i]}'",
                    )

            # Check for extra parameters
            extra_params = set(actual_param_names) - set(expected_param_names)
            if extra_params:
                self.add_violation(
                    component, f"Unexpected parameters: {', '.join(extra_params)}"
                )

            # Check for missing parameters
            missing_params = set(expected_param_names) - set(actual_param_names)
            if missing_params:
                self.add_violation(
                    component, f"Missing parameters: {', '.join(missing_params)}"
                )

            # Check each parameter's properties
            for param_name, param_info in expected_params.items():
                if param_name not in params:
                    continue  # Already reported as missing

                actual_param = params[param_name]

                # Check if optional/required matches
                is_optional = param_info.get("optional", False)
                has_default = actual_param.default != inspect.Parameter.empty

                if not is_optional and has_default:
                    # Check if default is None (which makes it effectively optional)
                    if actual_param.default is not None:
                        self.add_violation(
                            component,
                            f"Parameter {param_name} should be required but has default value",
                        )
                elif is_optional and not has_default:
                    self.add_violation(
                        component,
                        f"Parameter {param_name} should be optional but has no default value",
                    )

                # Check default value if specified
                if "default" in param_info and has_default:
                    expected_default = param_info["default"]
                    actual_default = actual_param.default
                    if actual_default != expected_default:
                        self.add_violation(
                            component,
                            f"Parameter {param_name} default value mismatch: "
                            f"expected {expected_default!r}, got {actual_default!r}",
                        )

                # Check type hints if available
                expected_type = param_info.get("type")
                if expected_type:
                    try:
                        type_hints = get_type_hints(func)
                        if param_name in type_hints:
                            actual_type = type_hints[param_name]
                            # Basic type checking (this could be more sophisticated)
                            if not self._types_compatible(actual_type, expected_type):
                                self.add_violation(
                                    component,
                                    f"Parameter {param_name} type mismatch: expected {expected_type}, got {actual_type}",
                                )
                    except Exception:
                        # Type hints might not be available
                        pass

            # Check return type if specified
            if expected_return:
                try:
                    type_hints = get_type_hints(func)
                    if "return" in type_hints:
                        actual_return = type_hints["return"]
                        if not self._types_compatible(actual_return, expected_return):
                            self.add_violation(
                                component,
                                f"Return type mismatch: expected {expected_return}, got {actual_return}",
                            )
                except Exception:
                    # Type hints might not be available
                    pass

        except Exception as e:
            self.add_violation(component, f"Error checking signature: {e}")

    def _types_compatible(self, actual, expected):
        """Check if two types are compatible, including complex generic types."""
        import typing

        # Direct match
        if actual == expected:
            return True

        # Get origins for generic types
        actual_origin = typing.get_origin(actual)
        expected_origin = typing.get_origin(expected)

        # Handle None type
        if expected is None or actual is None:
            return expected == actual

        # Handle Optional[X] which is Union[X, None]
        if expected_origin is Union:
            expected_args = typing.get_args(expected)
            # Check if actual matches any of the union members
            return any(self._types_compatible(actual, arg) for arg in expected_args)

        if actual_origin is Union:
            actual_args = typing.get_args(actual)
            # For Optional, check if expected is in the union
            if type(None) in actual_args:  # It's Optional
                # Remove None from args and check
                non_none_args = [arg for arg in actual_args if arg is not type(None)]
                if len(non_none_args) == 1:
                    return self._types_compatible(non_none_args[0], expected)
            # Check if expected is one of the union options
            return expected in actual_args

        # Handle List, Dict, etc.
        if actual_origin and expected_origin:
            # Both are generic types
            if actual_origin != expected_origin:
                return False

            # Check type arguments
            actual_args = typing.get_args(actual)
            expected_args = typing.get_args(expected)

            if len(actual_args) != len(expected_args):
                return False

            # Recursively check each type argument
            return all(
                self._types_compatible(a, e) for a, e in zip(actual_args, expected_args)
            )

        # Handle case where one is generic and other is not
        if actual_origin or expected_origin:
            # Special case: List vs list, Dict vs dict, etc.
            if actual_origin and actual_origin in (list, dict, tuple, set):
                return actual_origin == expected
            if expected_origin and expected_origin in (list, dict, tuple, set):
                return expected_origin == actual
            return False

        # For non-generic types, check if actual is a subclass of expected
        try:
            if isinstance(expected, type) and isinstance(actual, type):
                return issubclass(actual, expected)
        except TypeError:
            pass

        return False

    def check_module_exports(self, module_name: str, expected_exports: List[str]):
        """Check if a module exports the expected items."""
        try:
            module = __import__(module_name)
            for export in expected_exports:
                if not hasattr(module, export):
                    self.add_violation(
                        f"{module_name}.__all__", f"Missing export: {export}"
                    )
        except ImportError as e:
            self.add_violation(module_name, f"Module import error: {e}")


# ============================================================================
# Individual test functions for 100% coverage of the public API
# ============================================================================


def test_contract_module_exports():
    """Test that the module exports match the contract."""
    tester = APIContractTester()

    tester.check_module_exports(
        "cyborgdb",
        [
            "Client",
            "EncryptedIndex",
            "IndexConfig",
            "IndexIVF",
            "IndexIVFPQ",
            "IndexIVFFlat",
            "CyborgVectorStore",  # Optional, might fail if langchain not installed
        ],
    )

    # CyborgVectorStore is optional, so don't fail if it's missing
    violations = [v for v in tester.violations if "CyborgVectorStore" not in str(v)]
    assert len(violations) == 0, (
        f"Module export violations: {[str(v) for v in violations]}"
    )


def test_contract_client_class():
    """Test that the Client class API matches the contract."""
    tester = APIContractTester()

    Client = tester.check_class_exists("cyborgdb", "Client")
    if Client:
        # Check all Client methods
        tester.check_function_signature(
            Client,
            "__init__",
            expected_params={
                "base_url": {"type": str, "optional": False},
                "api_key": {"type": str, "optional": False},
                "verify_ssl": {"type": bool, "optional": True, "default": None},
            },
        )

        tester.check_function_signature(
            Client,
            "generate_key",
            expected_params={},
            expected_return=bytes,
            is_static=True,
        )

        tester.check_function_signature(
            Client, "get_health", expected_params={}, expected_return=Dict[str, str]
        )

        tester.check_function_signature(
            Client, "list_indexes", expected_params={}, expected_return=List[str]
        )

        tester.check_function_signature(
            Client,
            "create_index",
            expected_params={
                "index_name": {"type": str, "optional": False},
                "index_key": {"type": bytes, "optional": False},
                "index_config": {"optional": True, "default": None},
                "embedding_model": {"type": str, "optional": True, "default": None},
                "metric": {"type": str, "optional": True, "default": None},
            },
        )

        tester.check_function_signature(
            Client,
            "load_index",
            expected_params={
                "index_name": {"type": str, "optional": False},
                "index_key": {"type": bytes, "optional": False},
            },
        )

    assert len(tester.violations) == 0, (
        f"Client class contract violations: {[str(v) for v in tester.violations]}"
    )


def test_contract_encrypted_index_class():
    """Test that the EncryptedIndex class API matches the contract."""
    tester = APIContractTester()

    EncryptedIndex = tester.check_class_exists("cyborgdb", "EncryptedIndex")
    if EncryptedIndex:
        # Check properties
        tester.check_function_signature(
            EncryptedIndex,
            "index_name",
            expected_params={},
            expected_return=str,
            is_property=True,
        )

        tester.check_function_signature(
            EncryptedIndex,
            "index_type",
            expected_params={},
            expected_return=str,
            is_property=True,
        )

        tester.check_function_signature(
            EncryptedIndex,
            "index_config",
            expected_params={},
            expected_return=Dict[str, Any],
            is_property=True,
        )

        # Check methods
        tester.check_function_signature(
            EncryptedIndex, "is_trained", expected_params={}, expected_return=bool
        )

        tester.check_function_signature(
            EncryptedIndex, "is_training", expected_params={}, expected_return=bool
        )

        tester.check_function_signature(
            EncryptedIndex, "delete_index", expected_params={}, expected_return=None
        )

        tester.check_function_signature(
            EncryptedIndex,
            "upsert",
            expected_params={
                "arg1": {"optional": False},
                "arg2": {"optional": True},
            },
            expected_return=None,
        )

        tester.check_function_signature(
            EncryptedIndex,
            "delete",
            expected_params={
                "ids": {"type": List[str], "optional": False},
            },
            expected_return=None,
        )

        tester.check_function_signature(
            EncryptedIndex,
            "get",
            expected_params={
                "ids": {"type": List[str], "optional": False},
                "include": {
                    "type": List[str],
                    "optional": True,
                    "default": ["vector", "contents", "metadata"],
                },
            },
            expected_return=List[Dict[str, Any]],
        )

        tester.check_function_signature(
            EncryptedIndex, "list_ids", expected_params={}, expected_return=List[str]
        )

        tester.check_function_signature(
            EncryptedIndex,
            "query",
            expected_params={
                "query_vectors": {"optional": True},
                "query_contents": {"type": str, "optional": True},
                "top_k": {"type": int, "optional": True},
                "n_probes": {"type": int, "optional": True},
                "filters": {"type": Dict[str, Any], "optional": True},
                "include": {"type": List[str], "optional": True},
                "greedy": {"type": bool, "optional": True},
            },
            expected_return=Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]],
        )

        tester.check_function_signature(
            EncryptedIndex,
            "train",
            expected_params={
                "n_lists": {"type": int, "optional": True, "default": None},
                "batch_size": {"type": int, "optional": True, "default": None},
                "max_iters": {"type": int, "optional": True, "default": None},
                "tolerance": {"type": float, "optional": True, "default": None},
            },
            expected_return=None,
        )

    assert len(tester.violations) == 0, (
        f"EncryptedIndex class contract violations: {[str(v) for v in tester.violations]}"
    )


def test_contract_index_configs():
    """Test that the index configuration classes are properly exported and instantiable."""
    tester = APIContractTester()

    # Check IndexIVF
    IndexIVF = tester.check_class_exists("cyborgdb", "IndexIVF")
    if IndexIVF:
        try:
            _ = IndexIVF()  # Should work with defaults
        except Exception as e:
            tester.add_violation(
                "IndexIVF.__init__", f"Cannot instantiate with defaults: {e}"
            )

    # Check IndexIVFFlat
    IndexIVFFlat = tester.check_class_exists("cyborgdb", "IndexIVFFlat")
    if IndexIVFFlat:
        try:
            _ = IndexIVFFlat()
        except Exception as e:
            tester.add_violation(
                "IndexIVFFlat.__init__", f"Cannot instantiate with defaults: {e}"
            )

    # Check IndexIVFPQ
    IndexIVFPQ = tester.check_class_exists("cyborgdb", "IndexIVFPQ")
    if IndexIVFPQ:
        try:
            _ = IndexIVFPQ()
        except Exception:
            # Try with required params
            try:
                _ = IndexIVFPQ(pq_dim=64, pq_bits=8)
            except Exception as e2:
                tester.add_violation("IndexIVFPQ.__init__", f"Cannot instantiate: {e2}")

    assert len(tester.violations) == 0, (
        f"Index config violations: {[str(v) for v in tester.violations]}"
    )


def test_contract_index_config():
    """Test that IndexConfig is properly exported."""
    tester = APIContractTester()

    IndexConfig = tester.check_class_exists("cyborgdb", "IndexConfig")
    assert IndexConfig is not None, "IndexConfig should be exported from cyborgdb"
    assert len(tester.violations) == 0, (
        f"IndexConfig violations: {[str(v) for v in tester.violations]}"
    )


def test_api_contract_summary():
    """Generate a summary of the current API contract for documentation."""
    import cyborgdb

    print("\n" + "=" * 80)
    print("CURRENT API CONTRACT SUMMARY")
    print("=" * 80)

    # List all public exports
    print("\nPublic exports from cyborgdb:")
    for item in dir(cyborgdb):
        if not item.startswith("_"):
            print(f"  - {item}")

    # Check Client methods
    if hasattr(cyborgdb, "Client"):
        print("\nClient class methods:")
        for method in dir(cyborgdb.Client):
            if not method.startswith("_") or method == "__init__":
                print(f"  - {method}")

    # Check EncryptedIndex methods
    if hasattr(cyborgdb, "EncryptedIndex"):
        print("\nEncryptedIndex class methods:")
        for method in dir(cyborgdb.EncryptedIndex):
            if not method.startswith("_"):
                print(f"  - {method}")

    print("=" * 80 + "\n")
