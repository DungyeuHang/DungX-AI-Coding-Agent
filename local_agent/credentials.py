from __future__ import annotations

from abc import ABC, abstractmethod

import keyring
from keyring.errors import NoKeyringError


class CredentialStore(ABC):
    """Abstract base class for a secure credential store."""
    @abstractmethod
    def save(self, service_id: str, username: str, secret: str) -> None:
        pass

    @abstractmethod
    def get(self, service_id: str, username: str) -> str | None:
        pass

    @abstractmethod
    def delete(self, service_id: str, username: str) -> None:
        pass

    def has(self, service_id: str, username: str) -> bool:
        return self.get(service_id, username) is not None


class KeyringCredentialStore(CredentialStore):
    """Uses the system's native keyring for storing secrets."""
    def save(self, service_id: str, username: str, secret: str) -> None:
        try:
            keyring.set_password(service_id, username, secret)
        except NoKeyringError as e:
            raise RuntimeError("No system keyring backend found. Please install a backend like 'SecretService' or 'keyrings.alt'.") from e

    def get(self, service_id: str, username: str) -> str | None:
        try:
            return keyring.get_password(service_id, username)
        except NoKeyringError:
            return None

    def delete(self, service_id: str, username: str) -> None:
        try:
            keyring.delete_password(service_id, username)
        except (keyring.errors.PasswordDeleteError, NoKeyringError):
            # Ignore if it doesn't exist or backend has issues deleting
            pass


class MockCredentialStore(CredentialStore):
    """An in-memory mock for testing purposes."""
    def __init__(self):
        self._store: dict[tuple[str, str], str] = {}

    def save(self, service_id: str, username: str, secret: str) -> None:
        self._store[(service_id, username)] = secret

    def get(self, service_id: str, username: str) -> str | None:
        return self._store.get((service_id, username))

    def delete(self, service_id: str, username: str) -> None:
        self._store.pop((service_id, username), None)