# Goal of this script is to connect to the sharpoint repository and print the folder structure 

from office365.sharepoint.client_context import ClientContext
from office365.runtime.auth.authentication_context import AuthenticationContext

site_url = 'https://dfkide.sharepoint.com/sites/Team_DS2024_Bosch_Data'
username = ''
password = ''

ctx_auth = AuthenticationContext(site_url)
if ctx_auth.acquire_token_for_user(username, password):
    ctx = ClientContext(site_url, ctx_auth)
    print("Authentication successful")
else:
    print("Authentication failed")

relative_path = '/sites/Team_DS2024_Bosch_Data/Freigegebene Dokumente/General/Data'

# Access to the folder
folder = ctx.web.get_folder_by_server_relative_url(relative_path)
folder.expand(["Files", "Folders"]).get().execute_query()

# Printing structure
def print_folder_structure(folder, indent=0):
    folder.expand(["Files", "Folders"]).get().execute_query()
    print(' ' * indent + folder.properties["Name"])
    for sub_folder in folder.folders:
        print_folder_structure(sub_folder, indent + 2)
    for file in folder.files:
        print(' ' * (indent + 2) + file.properties["Name"])

print_folder_structure(folder)