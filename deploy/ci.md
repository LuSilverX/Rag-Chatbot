# Automated releases

GitHub Actions runs pytest against PostgreSQL/pgvector on pull requests and
updates to `main`, then builds a Linux Docker image. Only a passing `main`
revision can reach the deployment job. The image and deployment configuration
are uploaded together; third-party actions are pinned to commit hashes.

Repository settings require:

- Actions variable `DEPLOY_URL`: `https://77.113.94.93`
- Actions secret `DEPLOY_TOKEN`: the random server token, never committed

The server's `rag-deploy.service` listens on its private Docker bridge address.
Nginx exposes only its authenticated `POST /_deploy/` route through HTTPS. The
receiver streams the upload to disk, checks archive paths and the image's commit
label, and serializes deployments with a lock. SSH stays restricted to the
administrator's IP. No permanent AWS keys or GitHub tokens are stored on AWS.

Each release saves the old image and configuration, applies migrations/static
files, starts the services, and checks Docker health plus public HTTPS health,
login and static files. Failure restores the old image/configuration and causes
the GitHub job to fail. The active image tag is saved in `.deploy-state/active.env`
so later maintenance commands use the deployed revision. Current and previous
images are retained; older commit-tagged images are removed after success.

Rollback covers application images and configuration. Database migrations must
remain compatible with the previous version; the deployer does not undo database
changes or restore old data automatically. Use additive migrations and separate
destructive schema changes from releases that remove old application behavior.

Inspect deployments with `sudo journalctl -u rag-deploy.service` and
`sudo ./deploy/public.sh ps`. The receiver and apply script are installed under
`/usr/local/lib/rag-deploy`, outside the files replaced by a deployment. Changes
to these bootstrap files require an administrator to reinstall and restart the
service. Keep `/etc/rag-portfolio/deploy-token` readable only by root and rotate
the matching GitHub secret whenever replacing it.
