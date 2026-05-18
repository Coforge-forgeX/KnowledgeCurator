"""
Unified JWT/SSO token verification for kbcurator.
Accepts both legacy JWTs and Azure AD SSO tokens.
"""
from .auth import verify_jwt_token as legacy_verify_jwt_token
from .azure_sso import verify_microsoft_token

from .constants import Role

def verify_token(token: str):
    """
    Try to verify as legacy JWT first. If it fails, try as Azure AD SSO token.
    When an Azure AD token is accepted, resolve the email to the database user_id
    so that all downstream tools receive an integer user_id in jwt_claims.
    Raises on failure.
    """
    try:
        return legacy_verify_jwt_token(token)
    except Exception:
        # Try as Azure AD SSO token
        ms_claims = verify_microsoft_token(token)

        # Resolve Microsoft identity to our DB user so tools get an integer user_id
        email = (
            ms_claims.get("preferred_username")
            or ms_claims.get("upn")
            or ms_claims.get("email")
            or ""
        ).strip().lower()

        if email:
            from .auth import _fetch_user_by_email
            try:
                user = _fetch_user_by_email(email)
                role_id = user.get("role_id")
                if user:
                    ms_claims["user_id"] = user["user_id"]
                    ms_claims["email"] = user["email_id"]
                    ms_claims["is_admin"] = True if (role_id == Role.ADMIN.id) else False
                    ms_claims["role_id"] = role_id
            except Exception:
                pass

        return ms_claims
