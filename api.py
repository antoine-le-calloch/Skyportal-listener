import requests

class SkyPortal:
    """
    SkyPortal API client

    Parameters
    ----------
    protocol : str
        Protocol to use (http or https)
    host : str
        Hostname of the SkyPortal instance
    port : int
        Port to use
    token : str
        SkyPortal API token
    validate : bool, optional
        If True, validate the SkyPortal instance and token
    
    Attributes
    ----------
    base_url : str
        Base URL of the SkyPortal instance
    headers : dict
        Authorization headers to use
    """

    def __init__(self, instance, port, token, validate=True):
        # build the base URL from the protocol, host, and port
        self.base_url = f'{instance}'
        if port not in ['None', '', 80, 443]:
            self.base_url += f':{port}'
        
        self.headers = {'Authorization': f'token {token}'}

        # ping it to make sure it's up, if validate is True
        if validate:
            if not self._ping(self.base_url):
                raise ValueError('SkyPortal API not available')
            
            if not self._auth(self.base_url, self.headers):
                raise ValueError('SkyPortal API authentication failed. Token may be invalid.')
            
    def _ping(self, base_url):
        """
        Check if the SkyPortal API is available
        
        Parameters
        ----------
        base_url : str
            Base URL of the SkyPortal instance
            
        Returns
        -------
        bool
            True if the API is available, False otherwise
        """
        response = requests.get(f"{base_url}/api/sysinfo")
        return response.status_code == 200
    
    def _auth(self, base_url, headers):
        """
        Check if the SkyPortal Token provided is valid

        Parameters
        ----------
        base_url : str
            Base URL of the SkyPortal instance
        headers : dict
            Authorization headers to use

        Returns
        -------
        bool
            True if the token is valid, False otherwise
        """
        response = requests.get(
            f"{base_url}/api/config",
            headers=headers
        )
        return response.status_code == 200

    def api(self, method: str, endpoint: str, data=None, return_raw=False):
        """
        Make an API request to SkyPortal

        Parameters
        ----------
        method : str
            HTTP method to use (GET, POST, PUT, PATCH, DELETE)
        endpoint : str
            API endpoint to query
        data : dict, optional
            JSON data to send with the request, as parameters or payload
        return_raw : bool, optional
            If True, return raw response text instead of JSON

        Returns
        -------
        int
            HTTP status code
        dict
            JSON response
        """
        endpoint = f'{self.base_url}/{endpoint.strip("/")}'
        if method == 'GET':
            response = requests.request(method, endpoint, params=data, headers=self.headers)
        else:
            response = requests.request(method, endpoint, json=data, headers=self.headers)

        if return_raw:
            return response.status_code, response.text

        try:
            body = response.json()
        except Exception:
            raise ValueError(f'Error parsing JSON response: {response.text}')
        
        return response.status_code, body

    
    def get_spectra(
            self,
            id: int = None,
            obj_id: str = None,
            instrument_ids: list[int] = None,
            group_ids: list[int] = None,
            modified_after: str = None,
            modified_before: str = None,
            minimal: bool = False
        ):
        """
        Get spectra from SkyPortal (see https://skyportal.io/docs/api.html#tag/spectra/paths/~1api~1spectra/get)

        Parameters
        ----------
        id : int, optional
            ID of the spectrum to retrieve. If None, query by other parameters
        obj_id : str, optional
            Object ID to filter by
        instrument_ids : list[int], optional
            List of instrument IDs to filter by
        group_ids : list[int], optional
            List of group IDs to filter by
        modified_after : str, optional
            Get spectra modified after this date
        modified_before : str, optional
            Get spectra modified before this date
        minimal : bool, optional
            If True, return minimal payload (metadata only)

        Returns
        -------
        int
            HTTP status code
        dict
            JSON response

        Raises
        ------
        ValueError
            If no query parameters are provided

        """
        
        endpoint = 'api/spectra'
        data = {}
        if id:
            return self.api('GET', f'{endpoint}/{id}')
        
        if obj_id:
            data['objID'] = obj_id
        if instrument_ids:
            data['instrumentIDs'] = ','.join(map(str, instrument_ids))
        if group_ids:
            data['groupIDs'] = ','.join(map(str, group_ids))
        if modified_before:
            data['modifiedBefore'] = modified_before
        if modified_after:
            data['modifiedAfter'] = modified_after
        if minimal:
            data['minimalPayload'] = minimal

        if len(data) == 0:
            raise ValueError('No query parameters provided, specify at least one')
        response = self.api('GET', endpoint, data)
        return response


    def get_photometry(self, obj_id: str = None):
        """
        Get photometry from SkyPortal

        Parameters
        ----------
        obj_id : str
            Object ID to retrieve photometry for

        Returns
        -------
        int
            HTTP status code
        dict
            JSON response

        """
        endpoint = f"api/sources/{obj_id}/photometry"
        response = self.api('GET', endpoint)
        return response
