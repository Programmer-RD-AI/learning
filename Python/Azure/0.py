import pyodbc
import textwrap

# Specify the Driver.

driver = "{ODBC Driver 17 for SQL Server}"

# Specify the Server name and the DB name

server_name = "programmer-rd-ai"
database_name = "programmerrdai"


# Password - Programmer-RD-AI
# Server admin login = ranuga2008
server = f"{server_name}.database.windows.net,1433"

# Define User Name
username = "ranuga2008"
password = "Programmer-RD-AI"

# Create the full connection str

connection_str = textwrap.dedent(
    f"""
                                 Driver={driver};
                                 Server={server};
                                 Database={database_name};
                                 Uid={username};
                                 Pwd={password};
                                 Encrypt=yes;
                                 TrustServerCertificate=no;
                                 Connection Timeout=30;
                                 """
)
print(connection_str)

# Create a new PYOCDC connection object

cnxn: pyodbc.Connection = pyodbc.connect(connection_str)

# Create a new Cursor from the connection

crsr: pyodbc.Cursor = cnxn.cursor()

# Dine a select query
select_sql = "CREATE TABLE TEST (A varchar(50),B varchar(50))"
result = crsr.execute(select_sql)
select_sql = """
INSERT INTO
    [TEST](
        [A],
        [B]
    )
VALUES
    (
        2,
        'Jane'
    )
"""
result = crsr.execute(select_sql)
select_sql = "SELECT * FROM TEST"
result = crsr.execute(select_sql)
print(crsr.fetchall())

cnxn.close()
