import pymysql as m

con = m.connect(host = '54.164.15.159', db = 'team06-antifragile-db',
                user = 'admin', password = 'antifragile1234', charset = 'utf8')

cur = con.cursor()