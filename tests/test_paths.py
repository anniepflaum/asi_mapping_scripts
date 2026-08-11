import importlib


def reload_paths():
    import core.paths

    return importlib.reload(core.paths)


def test_explicit_image_and_output_roots(monkeypatch, tmp_path):
    images = tmp_path / "images"
    outputs = tmp_path / "mapped"
    monkeypatch.setenv("ASI_IMAGE_ROOT", str(images))
    monkeypatch.setenv("ASI_OUTPUT_ROOT", str(outputs))

    paths = reload_paths()

    assert paths.IMAGE_DIR == images
    assert paths.MAPPED_DIR == outputs


def test_lab_output_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ASI_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("LAB317_OUTPUT_ROOT", str(tmp_path))

    assert reload_paths().MAPPED_DIR == tmp_path / "asi-mapping" / "mapped"


def test_shared_data_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("ASI_IMAGE_ROOT", raising=False)
    monkeypatch.delenv("ASI_STARMAP_ROOT", raising=False)
    monkeypatch.delenv("ASI_TRAJECTORY_ROOT", raising=False)
    monkeypatch.delenv("ASI_RECEIVERS_PATH", raising=False)
    monkeypatch.setenv("LAB317_DATA_ROOT", str(tmp_path))

    paths = reload_paths()

    assert paths.IMAGE_DIR == tmp_path / "raw" / "asi" / "images"
    assert paths.STARMAP_DIR == tmp_path / "reference" / "starmaps"
    assert paths.TRAJECTORY_DIR == tmp_path / "raw" / "rocket" / "trajectories"
    assert paths.RECEIVERS_PATH == tmp_path / "reference" / "asi-mapping" / "receivers.csv"
