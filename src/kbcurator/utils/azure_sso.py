"""
Azure AD SSO token verification logic migrated from SSO service.
This module exposes verify_microsoft_token(token: str) for use in kbcurator.
"""
import os
import time
import certifi
import ssl
import jwt
from jwt import PyJWKClient
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

TENANT_ID: str = os.getenv("TENANT_ID", "").strip()
AUDIENCE: str = os.getenv("AUDIENCE", "").strip()
REQUIRED_SCOPE: str = os.getenv("REQUIRED_SCOPE", "access_as_user").strip()
ALLOWED_ROLE: str = os.getenv("ALLOWED_ROLE", "").strip()

OPENID_CONFIG_URL = (
    f"https://login.microsoftonline.com/{TENANT_ID}/v2.0/.well-known/openid-configuration"
)
OPENID_CONFIG_URL_V1 = (
    f"https://login.microsoftonline.com/{TENANT_ID}/.well-known/openid-configuration"
)

_ISSUER_V2: Optional[str] = None
_JWK_CLIENT_V2: Optional[PyJWKClient] = None
_META_FETCHED_AT_V2: float = 0.0
_ISSUER_V1: Optional[str] = None
_JWK_CLIENT_V1: Optional[PyJWKClient] = None
_META_FETCHED_AT_V1: float = 0.0
_META_TTL: int = 3600

def _load_oidc_config(config_url: str) -> tuple:
    _ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    import requests
    r = requests.get(config_url, timeout=10, verify=certifi.where())
    r.raise_for_status()
    meta = r.json()
    issuer = meta.get("issuer")
    jwks_uri = meta.get("jwks_uri")
    if not issuer or not jwks_uri:
        raise RuntimeError(f"OpenID metadata from {config_url} missing 'issuer' or 'jwks_uri'.")
    return issuer, PyJWKClient(jwks_uri, ssl_context=_ssl_ctx)

def _ensure_v2_loaded() -> None:
    global _ISSUER_V2, _JWK_CLIENT_V2, _META_FETCHED_AT_V2
    now = time.time()
    if _ISSUER_V2 and _JWK_CLIENT_V2 and (now - _META_FETCHED_AT_V2) < _META_TTL:
        return
    _ISSUER_V2, _JWK_CLIENT_V2 = _load_oidc_config(OPENID_CONFIG_URL)
    _META_FETCHED_AT_V2 = now

def _ensure_v1_loaded() -> None:
    global _ISSUER_V1, _JWK_CLIENT_V1, _META_FETCHED_AT_V1
    now = time.time()
    if _ISSUER_V1 and _JWK_CLIENT_V1 and (now - _META_FETCHED_AT_V1) < _META_TTL:
        return
    _ISSUER_V1, _JWK_CLIENT_V1 = _load_oidc_config(OPENID_CONFIG_URL_V1)
    _META_FETCHED_AT_V1 = now

def _peek_token(ms_token: str) -> dict:
    try:
        return jwt.decode(ms_token, options={"verify_signature": False})
    except Exception:
        return {}

def verify_microsoft_token(ms_token: str) -> Dict[str, Any]:
    peek = _peek_token(ms_token)
    token_ver = peek.get("ver", "2.0")
    token_aud = peek.get("aud", "unknown")
    token_iss = peek.get("iss", "unknown")
    GRAPH_APP_ID = "00000003-0000-0000-c000-000000000000"
    allowed_audiences = [AUDIENCE, GRAPH_APP_ID]
    if token_ver == "1.0":
        _ensure_v1_loaded()
        jwk_client = _JWK_CLIENT_V1
        expected_issuer = _ISSUER_V1
    else:
        _ensure_v2_loaded()
        jwk_client = _JWK_CLIENT_V2
        expected_issuer = _ISSUER_V2
    try:
        signing_key = jwk_client.get_signing_key_from_jwt(ms_token).key
    except Exception as e:
        raise ValueError(f"Key lookup failed: {e}. Token info → ver={token_ver}, aud={token_aud}, iss={token_iss}")
    try:
        claims = jwt.decode(
            ms_token,
            signing_key,
            algorithms=["RS256"],
            audience=allowed_audiences,
            issuer=expected_issuer,
            options={
                "require": ["iss", "aud", "exp"],
                "verify_signature": True,
                "verify_aud": True,
                "verify_exp": True,
                "verify_nbf": True,
            },
        )
    except jwt.ExpiredSignatureError:
        raise ValueError("Token expired.")
    except jwt.InvalidAudienceError:
        raise ValueError(f"Audience mismatch. Token aud='{token_aud}', allowed audiences='{allowed_audiences}'. Ensure the frontend requests a token scoped for YOUR API or Microsoft Graph.")
    except jwt.InvalidIssuerError:
        raise ValueError(f"Issuer mismatch. Token iss='{token_iss}', expected='{expected_issuer}'.")
    except jwt.PyJWTError as e:
        raise ValueError(f"Signature verification failed. Token info → ver={token_ver}, aud={token_aud}, iss={token_iss}. Detail: {e}")
    scopes = claims.get("scp") or ""
    roles = claims.get("roles") or []
    has_scope = REQUIRED_SCOPE in scopes.split() if REQUIRED_SCOPE else True
    has_role = ALLOWED_ROLE in roles if ALLOWED_ROLE else False
    if ALLOWED_ROLE:
        if not (has_scope or has_role):
            raise PermissionError(f"Missing scope '{REQUIRED_SCOPE}' or role '{ALLOWED_ROLE}'.")
    else:
        if not has_scope:
            raise PermissionError(f"Missing scope '{REQUIRED_SCOPE}'.")
    return claims
