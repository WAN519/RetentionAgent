import mysql.connector
from mysql.connector import Error
import os
import tempfile
from dotenv import load_dotenv



class DatabaseManager:

    def __init__(self):
        # load db_config
        load_dotenv('config.env')
        # Database configuration
        ssl_ca_content = os.environ.get('DB_SSL_CA_CONTENT')
        if ssl_ca_content:

            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            temp_file.write(ssl_ca_content)
            temp_file.close()
        # get db_config KEY
        DB_CONFIG = {
            'host': os.environ.get('DB_HOST'),
            'database': os.environ.get('DB_DATABASE'),
            'user': os.environ.get('DB_USER'),
            'password': os.environ.get('DB_PASSWORD'),
            'port': int(os.environ.get('DB_PORT', 5432)),
            'ssl_ca': temp_file.name,
            'ssl_verify_cert': bool(os.environ.get('DB_SSL_VERIFY_CERT')) == True
        }
        self.config = DB_CONFIG.copy()
        # add config for aiven
        self.config.update({
            'use_pure': True,
            'autocommit': False,
            'pool_name': 'mypool',
            'pool_size': 3
        })

    def get_connection(self):
        """get new connection"""
        try:
            return mysql.connector.connect(**self.config)
        except Error as e:
            print(f"connect fail: {e}")
            return None

    def connect(self):
        conn = self.get_connection()
        if conn and conn.is_connected():
            print("database connect successful")
            conn.close()
            return True
        return False

    def execute_sql(self, sql, params=None):
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return False

            cursor = conn.cursor()

            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)

            # Confirm and save all changes
            conn.commit()
            return True

        except Error as e:
            print(f"execute fail: {e}")
            print(f"SQL: {sql[:100]}...")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            return False

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def fetch_all(self, query, params=None):
        """all result"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return None

            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = cursor.fetchall()
            return result

        except Error as e:
            print(f"query fail: {e}")
            return None

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def fetch_one(self, query, params=None):
        """just one result"""
        conn = None
        cursor = None
        try:
            conn = self.get_connection()
            if not conn:
                return None

            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            result = cursor.fetchone()
            return result

        except Error as e:
            print(f"query fail: {e}")
            return None

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if conn:
                try:
                    conn.close()
                except:
                    pass

    def close(self):
        print("database operations completed")
