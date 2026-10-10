import importlib.util
import io
from pathlib import Path
import sys
import tarfile
from unittest.mock import patch
import pytest
from deploy import apply_release as release


TAG = 'a' * 40


def configure(root, value):
    for name in release.FILES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    (root / 'image.tar.gz').write_bytes(b'image')


def fake_command(args, **kwargs):
    if args[:2] == ['docker', 'inspect']:
        return 'rag-chatbot:local\n'
    if args[:3] == ['docker', 'image', 'inspect']:
        return TAG
    if args[:3] == ['docker', 'image', 'ls']:
        return TAG + '\nlocal\n'
    return ''


def test_failed_health_restores_image_and_configuration(tmp_path):
    app, candidate = tmp_path / 'app', tmp_path / 'candidate'
    configure(app, 'previous')
    configure(candidate, 'new')
    with patch.object(release, 'command', side_effect=fake_command), \
         patch.object(release, 'compose', return_value='container-id'), \
         patch.object(release, 'healthy', side_effect=[RuntimeError('unhealthy'), None]) as health:
        with pytest.raises(RuntimeError, match='previous image and configuration restored'):
            release.deploy(candidate, TAG, app)
    assert [call.args[1] for call in health.call_args_list] == [TAG, 'local']
    assert all((app / name).read_text() == 'previous' for name in release.FILES)
    assert not (app / '.deploy-state/active.env').exists()


def test_success_persists_image_and_keeps_previous_configuration(tmp_path):
    app, candidate = tmp_path / 'app', tmp_path / 'candidate'
    configure(app, 'previous')
    configure(candidate, 'new')
    with patch.object(release, 'command', side_effect=fake_command), \
         patch.object(release, 'compose', return_value='container-id'), \
         patch.object(release, 'healthy'):
        result = release.deploy(candidate, TAG, app)
    assert result['revision'] == TAG
    assert (app / '.deploy-state/active.env').read_text() == f'APP_IMAGE_TAG={TAG}\n'
    assert (app / '.deploy-state/previous-config/compose.public.yml').read_text() == 'previous'


def test_wrong_image_revision_does_not_touch_running_configuration(tmp_path):
    app, candidate = tmp_path / 'app', tmp_path / 'candidate'
    configure(app, 'previous')
    configure(candidate, 'new')
    def wrong_revision(args, **kwargs):
        return 'b' * 40 if args[:3] == ['docker', 'image', 'inspect'] else fake_command(args)
    with patch.object(release, 'command', side_effect=wrong_revision), \
         patch.object(release, 'compose', return_value='container-id'), \
         patch.object(release, 'healthy') as health:
        with pytest.raises(ValueError, match='revision'):
            release.deploy(candidate, TAG, app)
    health.assert_not_called()
    assert (app / 'compose.public.yml').read_text() == 'previous'


def test_rollback_failure_is_reported(tmp_path):
    app, candidate = tmp_path / 'app', tmp_path / 'candidate'
    configure(app, 'previous')
    configure(candidate, 'new')
    with patch.object(release, 'command', side_effect=fake_command), \
         patch.object(release, 'compose', return_value='container-id'), \
         patch.object(release, 'healthy', side_effect=RuntimeError('unavailable')):
        with pytest.raises(RuntimeError, match='rollback needs attention'):
            release.deploy(candidate, TAG, app)


def test_archive_traversal_is_rejected(tmp_path):
    # Load the standalone receiver as the systemd service does.
    with patch.dict(sys.modules, {'apply_release': release}):
        spec = importlib.util.spec_from_file_location('receiver', Path('deploy/receiver.py'))
        receiver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(receiver)
    archive = tmp_path / 'malicious.tar'
    with tarfile.open(archive, 'w') as bundle:
        member = tarfile.TarInfo('../outside')
        member.size = 1
        bundle.addfile(member, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='release files'):
        receiver.extract_release(archive, tmp_path / 'candidate')
    assert not (tmp_path / 'outside').exists()
