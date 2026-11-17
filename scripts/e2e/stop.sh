#!/bin/bash
# Stops everything start.sh started. Matches by command line, so run it from a shell that is not itself named like them.
for pat in "uvicorn app.main" "celery -A app.workers" "next start" "next-server" "moto_server"; do
  for pid in $(pgrep -f "$pat"); do [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null; done
done
redis-cli shutdown nosave 2>/dev/null
exit 0
