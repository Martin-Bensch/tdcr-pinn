import logging

STATUS = "DEVELOPMENT"
USELOGFILE = 1

if STATUS == "DEVELOPMENT":
    from pytdcrsv.development import *
else:
    print("No loggin specification?")
