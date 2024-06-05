from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext

url = "https://dfkide.sharepoint.com/sites/Team_DS2024_Bosch_Data'"
ctx_auth = AuthenticationContext(url)
token = ctx_auth.acquire_token_for_user('s8mooste', 'esi18am!')
print ('Authentication successful')
ctx = ClientContext(url, ctx_auth)

# response = ctx.web.get_file_by_server_relative_url('/path/to/your/file').download('/path/to/save/file')
# ctx.execute_query()

# with open('/path/to/save/file', 'rb') as file:
    # for chunk in iter(lambda: file.read(1024), b''):
        # process(chunk)  # Replace with your data processing logic
