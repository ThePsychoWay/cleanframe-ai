// Helpers shared by the end-to-end scripts: paths, the database (SQLite or PostgreSQL, whatever DATABASE_URL says),
// and the S3 stand-in. Run through run.sh, which sets the environment.
const { execFileSync } = require("child_process");
const path = require("path");

const DIR = process.env.CF_E2E_DIR;
const VENV = process.env.CF_VENV;
if (!DIR || !VENV) throw new Error("Run the end-to-end scripts through scripts/e2e/run.sh so the environment is set.");

const python = (code, input) =>
  execFileSync(`${VENV}/bin/python`, ["-c", code], { cwd: `${process.env.CF_ROOT}/backend`, input: input ?? "", env: process.env }).toString().trim();

/** Runs one SQL statement; returns the first column of the first row as text ("" if none). */
const sql = (statement, params = {}) =>
  python(
    `import json, os, sys
from sqlalchemy import create_engine, text
params = json.load(sys.stdin)
with create_engine(os.environ["DATABASE_URL"]).begin() as c:
    r = c.execute(text(${JSON.stringify(statement)}), params)
    row = r.fetchone() if r.returns_rows else None
    print("" if row is None else row[0])`,
    JSON.stringify(params),
  );

/** Makes projects overdue for deletion: `where` is a SQL condition (default: all of them). */
const makeDue = (where = "1=1") =>
  python(
    `import os, datetime
from sqlalchemy import create_engine, text
e = create_engine(os.environ["DATABASE_URL"])
t = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
if e.dialect.name == "sqlite": t = t.replace(tzinfo=None)
with e.begin() as c: c.execute(text("update projects set expires_at = :t where ${where.replace(/"/g, '\\"')}"), {"t": t})`,
  );

const storageKeys = () =>
  python(`import boto3
s3 = boto3.client("s3", endpoint_url="http://localhost:5000", region_name="us-east-1")
print("\\n".join(o["Key"] for o in s3.list_objects_v2(Bucket="cleanframe").get("Contents", [])))`)
    .split("\n")
    .filter(Boolean);

module.exports = { DIR, VENV, media: path.join(DIR, "media"), python, sql, makeDue, storageKeys, log: (name) => require("fs").readFileSync(path.join(DIR, name), "utf8") };
