# config.py
import os
import logging

# where to write images/models/csv
basepath = os.getenv("EPN_BASEPATH", "./")

# optional: define the custom logging level 21 used in your logs
LOG_DETAIL = 21
if not hasattr(logging, "DETAIL"):
    logging.addLevelName(LOG_DETAIL, "DETAIL")
    def detail(self, message, *args, **kws):
        if self.isEnabledFor(LOG_DETAIL):
            self._log(LOG_DETAIL, message, args, **kws)
    logging.Logger.detail = detail  # type: ignore
