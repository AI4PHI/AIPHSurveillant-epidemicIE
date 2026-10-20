"""Template configuration for the epidemiological information-extraction pipeline.

Copy this file to ``config.py`` and fill in your own values. ``config.py`` is gitignored and must
never be committed: it holds the credentials used to authenticate against the WHO/EIOS API.

The EIOS API uses Azure AD OAuth2 client-credentials authentication. Request access to the EIOS
platform and obtain these values from your EIOS administrator (see
https://eios.who.int/portal/api/swagger/index.html).
"""

# Azure AD tenant id used by EIOS token acquisition.
eios_tenant_id = "<your-eios-tenant-id>"

# OAuth2 client id (the consumer application id registered for your EIOS access).
eios_client_id = "<your-eios-client-id>"

# OAuth2 client secret. Keep this secret; do not share or commit it.
eios_client_secret = "<your-eios-client-secret>"

# EIOS API scope (documented as EIOS_CLIENT_ID_SCOPE), typically of the form
# "<api-app-id>/.default".
eios_scope = "<your-eios-api-scope>"
