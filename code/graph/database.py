from pathlib import Path
from typing import Optional
import sqlite3

import pandas as pd


class Database:

    """Database class to perform all functions associated with the OpenWPM crawl DB. 
    """

    def __init__(self, database_filename: Path):
        self.database_filename = database_filename
        self.conn: Optional[sqlite3.Connection] = None


    def __enter__(self):
        self.conn = sqlite3.connect(str(self.database_filename.resolve()))
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn is not None:
            self.conn.close()
        self.conn = None


    def website_from_visit_id(self, visit_id):

        """
        Function to get relevant table data for a particular website. 
        This data is ued to build graphs.

        Args:
            visit_id: Visit ID of the website we want.
        Returns:
            df_requests: DataFrame representation of requests table in OpenWPM.
            df_responses: DataFrame representation of responses table in OpenWPM.
            df_redirects: DataFrame representation of redirects table in OpenWPM.
            call_stacks: DataFrame representation of call_stacks table in OpenWPM.
            javascript: DataFrame representation of javascript table in OpenWPM.
        """

        if self.conn is None:
            raise sqlite3.ProgrammingError("Database not open")

        df_http_requests = pd.read_sql_query(
            "SELECT visit_id, request_id, "
            "url, headers, top_level_url, resource_type, "
            f"time_stamp, post_body, post_body_raw from http_requests where {visit_id} = visit_id",
            self.conn
        )
        df_http_responses = pd.read_sql_query(
            "SELECT visit_id, request_id, "
            "url, headers, response_status, time_stamp, content_hash "
            f" from http_responses where {visit_id} = visit_id",
            self.conn
        )
        df_http_redirects = pd.read_sql_query(
            "SELECT visit_id, old_request_id, "
            "old_request_url, new_request_url, response_status, "
            f"headers, time_stamp from http_redirects where {visit_id} = visit_id",
            self.conn
        )
        call_stacks = pd.read_sql_query(
            f"SELECT visit_id, request_id, call_stack from callstacks where {visit_id} = visit_id",
            self.conn
        )
        javascript = pd.read_sql_query(
            "SELECT visit_id, script_url, script_line, script_loc_eval, top_level_url, document_url, symbol, call_stack, operation,"
            f" arguments, attributes, value, time_stamp from javascript where {visit_id} = visit_id",
            self.conn
        )
        return df_http_requests, df_http_responses, df_http_redirects, call_stacks, javascript


    def sites_visits(self):

        """
        Function to get site visit table data.

        Returns:
            DataFrame representation of the site_visits table for successfully crawled sites.
        """

        df_successful_sites = pd.read_sql_query(
            "SELECT visit_id from crawl_history where "
            "command = 'GetCommand' and command_status = 'ok'",
            self.conn
        )

        successful_vids = df_successful_sites['visit_id'].tolist()

        return pd.read_sql_query(
            f"SELECT visit_id, site_url from site_visits where visit_id in ({','.join([str(x) for x in tuple(successful_vids)])})",
            self.conn
        )
