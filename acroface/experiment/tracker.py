import csv
import itertools
import shutil
import copy
from collections import defaultdict
from dataclasses import dataclass, is_dataclass
from typing import Optional, Mapping
import json
from pathlib import Path, PurePath
import datetime
import os.path
import shutil
import tempfile

import numpy as np
import pandas as pd
import dill


DEFAULT_CHILD_GROUP = 'children'

class JSONEncoder(json.JSONEncoder):
    "Custom JSONEncoder which tries to encode filed types (like pathlib Paths) as strings"
    def default(self, o):
        if is_dataclass(o):
            attributes = copy.copy(o.__dict__)
            attributes['dataclass_name'] = o.__class__.__name__
            attributes['dataclass_module'] = o.__module__
            return attributes
        try:
            return json.JSONEncoder.default(self, o)
        except TypeError:
            return str(o)


def atomic_write():
    pass


@dataclass
class Event:
    content: str
    id: int
    timestamp: datetime.datetime


class ExperimentTracker(object):
    def __init__(self, output_dir: Path, identifier=None, parent=None, tag=None, starting_count=0, reset_tracker=False):
        """
        Create an ExperimentTracker object. If the output_path is not an existing experiment, it will be freshly initialized.
        If it is an existing experiment, the state of the experiment tracker will be read from it.
        """
        self.output_dir = output_dir
    
        self.tracker_info_path = self.output_dir / 'tracker_info.json'
        self.event_log = self.output_dir / 'events.txt'
        self.metadata_path = self.output_dir / 'experiment_metadata'
        self.progress_path = self.output_dir / 'progress.txt'
        self.children_info_path = self.output_dir / 'children_info.json'
        
        self.children = defaultdict(list)
        
        if reset_tracker or not self.tracker_info_path.exists():
            # Initialize this tracker as a new one
            self.initialize_tracker(identifier=identifier, parent=parent, tag=tag, starting_progress=starting_count)
        else:
            self.restore_tracker()

        
    def initialize_tracker(self, *, identifier, parent, tag, starting_progress):
        self.parent = parent
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)
        self.metadata_path.mkdir(parents=True, exist_ok=True)
        
        self.identifier = identifier
        self.tag = tag
        
        self.tracker_info = dict(identifier=self.identifier, tag=self.tag)
        with open(self.tracker_info_path, 'w') as fp:
            json.dump(self.tracker_info, fp)
        
        with open(self.children_info_path, 'w') as fp:
            json.dump(dict(), fp)
        
        with open(self.event_log, 'w') as fp:
            pass
        
        # These are the stateful entries        
        self.event_id = 0
        self.set_progress(starting_progress)
    
    def restore_tracker(self):
        with open(self.tracker_info_path) as fp:
            self.tracker_info = json.load(fp)
        self.identifier = self.tracker_info['identifier']
        self.tag = self.tracker_info['tag']
        
        # Find the experiment progress
        with open(self.progress_path) as fp:
            for line in fp:
                try:
                    progress = int(line)
                except ValueError:
                    break
            self.set_progress(progress)
            
        # Find the children
        with open(self.children_info_path) as fp:
            children_info = json.load(fp)
            for child_group, child_entries in children_info.items():
                for (relpath, child_tag) in child_entries:
                    child_path = self.output_dir / relpath
                    self.children[child_group].append((child_path, child_tag))
        
        # Determine the events
        with open(self.event_log) as fp:
            event_id = None
            for line in fp:
                event_id, _ = line.strip().split()
            if event_id is not None:
                self.event_id = int(event_id) + 1
            else:
                self.event_id = 0
        

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def log_event(self, event: str, reference_event: Optional[Event] = None) -> Event:
        t = datetime.datetime.now().replace(microsecond=0)
        event_id = self.event_id
        self.event_id += 1
        event = Event(content=event, id=event_id, timestamp=t)
        #self.events[event_id] = event
        
        with open(self.event_log, 'a') as fp:
            timestamp = t.isoformat()
            if reference_event is not None:
                timedelta = t - reference_event.timestamp
                # We remove the microseconds for clarity
                timedelta = str(datetime.timedelta(seconds=round(timedelta.total_seconds())))
                fp.write(f'{event_id:02} {timestamp} [{timedelta} since event {reference_event.id}]:\t{event.content}\n')
            else:
                fp.write(f'{event_id:02} {timestamp}:\t{event.content}\n')
            fp.flush()
        return event

    def set_flag(self, flag_name, flag_status=True):
        flags_directory = self.output_dir / 'flags'
        flags_directory.mkdir(exist_ok=True)
        flag_path = flags_directory / flag_name
        with open(flag_path, 'w') as fp:
            fp.write(str(flag_status))
    
    def check_flag(self, flag_name):
        flags_directory = self.output_dir / 'flags'
        flag_path = flags_directory / flag_name
        if not flag_path.exists():
            raise FileNotFoundError(f"Flag does not exist: {str(flag_path)}")
        with open(flag_path, 'r') as fp:
            flag_status = fp.read()
            return flag_status == str(True)

    def log_artifact(self, name, artifact):
        artifacts_path = self.output_dir / 'artifacts'
        artifacts_path.mkdir(exist_ok=True)
        with open(artifacts_path / f'{name}.pkl', 'wb') as fp:
            dill.dump(artifact, fp)

    def load_artifact(self, name):
        artifacts_path = self.output_dir / 'artifacts'
        with open(artifacts_path / f'{name}.pkl', 'rb') as fp:
            artifact = dill.load(fp)
            return artifact
        
    def log_artifacts(self, artifacts):
        if isinstance(artifacts, Mapping):
            artifacts = artifacts.items()
        for artifact_name, artifact in artifacts:
            self.log_artifact(artifact_name, artifact)

    def log_file(self, src_path: Path, dst_path=None):
        files_path = self.output_dir / 'files'
        files_path.mkdir(exist_ok=True)
        if dst_path is None:
            dst_path = src_path.name
        dst_path = files_path / dst_path
        shutil.copy(src_path, dst_path)

    def log_files(self, files):
        if isinstance(files, Mapping):
            files = [(src, dst) for dst, src in files.items()]
        for item in files:
            try:
                src, dst = item
                self.log_file(src, dst)
            except ValueError:
                self.log_file(item)

    def log_scalar(self, name, value):
        scalars_dir = self.output_dir / 'scalars'
        scalars_dir.mkdir(exist_ok=True)
        scalars_path = scalars_dir / (name + '.csv')
        if not scalars_path.exists():
            with open(scalars_path, 'a') as fp:
                fp.write('timestamp,value\n')
                fp.write(f'{self.experiment_progress},{value}\n')
        else:
            with open(scalars_path, 'a') as fp:
                fp.write(f'{self.experiment_progress},{value}\n')

    def log_scalars(self, scalars):
        if isinstance(scalars, Mapping):
            scalars = scalars.items()
        for name, value in scalars:
            self.log_scalar(name, value)

    def log_vector(self, name, value):
        raise NotImplementedError('log_vector has not been implemented')

    def log_model(self, model_reference, model_object):
        model_directory = self.output_dir / 'models'
        model_directory.mkdir(exist_ok=True)
        model_factory, model_state = model_object.serialize(model_directory, tag=model_reference)
        model_path = model_directory / (model_reference + '.pkl')
        with open(model_path, 'wb') as fp:
            dill.dump(dict(model_factory=model_factory,
                           model_state=model_state), fp)
        self.make_reference(model_reference, model_path)

    def log_json(self, name, value):
        jsons_dir = self.output_dir / 'jsons'
        jsons_dir.mkdir(exist_ok=True)
        json_file_path = jsons_dir / (name + '.json')
        with open(json_file_path, 'w') as fp:
            json.dump(value, fp, cls=JSONEncoder)

    def tablify_values(self, values):
        if isinstance(values, Mapping):
            try:
                values = pd.DataFrame(values)
            except ValueError:
                # For now we assume the problem is that each "column" is scalar
                values = pd.DataFrame({k: [v] for k, v in values.items()})
        return values

    def log_table(self, name, values):
        tables_path = self.output_dir / 'tables'
        tables_path.mkdir(exist_ok=True)
        csv_path = tables_path / (name + '.csv')
        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)

    def load_table(self, table_name):
        table_path = self.output_dir / 'tables' /(table_name + '.csv')
        table = pd.read_csv(table_path)
        return table
        
    def log_performance(self, dataset_name, values, tag=None):
        tables_path = self.output_dir / 'evaluation' / dataset_name / 'performance'
        tables_path.mkdir(exist_ok=True, parents=True)

        if tag is not None:
            csv_path = tables_path / f'{tag}_performance.csv'
        else:
            csv_path = tables_path / f'performance.csv'

        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)

    def log_predictions(self, dataset_name, values, tag=None):
        tables_path = self.output_dir / 'evaluation' / dataset_name / 'predictions'
        tables_path.mkdir(exist_ok=True, parents=True)

        if tag is not None:
            csv_path = tables_path / f'{tag}_predictions.csv'
        else:
            csv_path = tables_path / f'predictions.csv'

        values = self.tablify_values(values)
        if isinstance(values, pd.DataFrame):
            values.to_csv(csv_path, index=False)

    def log_numpy(self, name, value):
        numpy_dir = self.output_dir / 'ndarrays'
        numpy_dir.mkdir(exist_ok=True)

        if isinstance(value, dict):
            np.savez(numpy_dir / name, value)
        else:
            np.save(numpy_dir / name, value)

    def reference_exists(self, reference):
        "Check if a reference exists"
        references_dir = self.output_dir / 'references'
        references_path = references_dir / (reference + '.pkl')
        return references_path.exists()

    def lookup_reference(self, reference):
        references_dir = self.output_dir / 'references'
        references_path = references_dir / (reference + '.pkl')
        with open(references_path, 'rb') as fp:
            value = dill.load(fp)
            if isinstance(value, PurePath):
                if not value.is_absolute():
                    joined_path = Path(references_path / value).resolve()
                    if joined_path.exists():
                        value = joined_path
                    # If the value is a PurePath object, convert it to a regular 
                    # Path and if it's relative, try to resolve it
                value = Path(value)
            return value

    def delete_model(self, model_reference):
        model_path = self.lookup_reference(model_reference)
        try:
            model_path.unlink()
        except FileNotFoundError:
            pass
        self.remove_reference(model_reference)
        
    def purge_models(self):
        # TODO: It would be nice if we looked up the references and removed those as well.
        model_directory = self.output_dir / 'models'
        models = model_directory.glob('*.pkl')
        for model_path in models:
            model_path.unlink()

    def make_reference(self, reference_name, referred_value):
        references_dir = self.output_dir / 'references'
        references_dir.mkdir(exist_ok=True)
        references_path = references_dir / (reference_name + '.pkl')
        with open(references_path, 'wb') as fp:
            if isinstance(referred_value, PurePath):
                # Make any path reference relative to the reference pickle file
                referred_value = PurePath(os.path.relpath(referred_value, references_path))
            dill.dump(referred_value, fp)

    def remove_reference(self, reference_name):
        references_dir = self.output_dir / 'references'
        references_dir.mkdir(exist_ok=True)
        references_path = references_dir / (reference_name + '.pkl')
        try:
            references_path.unlink()
        except FileNotFoundError:
            pass

    def make_child(self, child_group: str=None, tag: str=None) -> "ExperimentTracker":
        if child_group is None:
            child_group = DEFAULT_CHILD_GROUP
        
        if tag is None:
            i = len(self.children[child_group])
            tag = f'{i:02}'
            
        # NOTE: if you change this, make sure you change the find_experiments function below
        children_dir = self.output_dir / child_group
        children_dir.mkdir(exist_ok=True)
        child_dir = children_dir / tag
        child_tracker = ExperimentTracker(child_dir, parent=self, tag=tag)
        self.children[child_group].append((child_dir, tag))
        with open(self.children_info_path, 'w') as fp:
            child_dump = {child_group: [(str(child_dir.relative_to(self.output_dir)), tag) for child_dir, tag in child_group_entries] for child_group, child_group_entries in self.children.items()}
            json.dump(child_dump, fp)
        return child_tracker
    
    def get_child(self, query_tag, child_group=DEFAULT_CHILD_GROUP):
        group_children = self.children[child_group]
        for child_path, child_tag in group_children:
            if child_tag == query_tag:
                return ExperimentTracker(child_path)
        raise KeyError(f"Child tracker {query_tag} in group {child_group} not found")

    def get_children(self, child_group=DEFAULT_CHILD_GROUP):
        child_trackers = []
        for child_path, child_tag in  self.children[child_group]:
            tracker = ExperimentTracker(child_path, parent=self, tag=child_tag)
            child_trackers.append(tracker)
        return child_trackers
    
    def get_children_paths(self, child_group=DEFAULT_CHILD_GROUP):
        child_paths = []
        for child_path, child_tag in  self.children[child_group]:
            child_paths.append(child_path)
        return child_paths
    
    def get_json(self, name):
        jsons_dir = self.output_dir / 'jsons'
        json_file_path = jsons_dir / (name + '.json')
        with open(json_file_path, 'r') as fp:
            value = json.load(fp)
        return value

    def load_model(self, reference, force_cpu=False):
        model_path = self.lookup_reference(reference)
        with open(model_path, 'rb') as fp:
            model_dict = dill.load(fp)
            model_factory = model_dict['model_factory']
            model_state = model_dict['model_state']
            model = model_factory(model_state)
            if force_cpu:
                model.set_device('cpu')
            return model
    
    def get_file_path(self, name):
        file_name = self.output_dir / 'files' / name
        return file_name

    def set_progress(self, progress):
        self.experiment_progress = progress
        with open(self.progress_path, 'a') as fp:
            fp.write(f"{self.experiment_progress}\n")
            
    def progress(self, n=1):
        self.experiment_progress += n
        with open(self.progress_path, 'a') as fp:
            fp.write(f"{self.experiment_progress}\n")

    def get_identifier(self):
        if self.identifier is not None:
            return self.identifier
        else:
            return self.output_dir.name

    def get_parent(self):
        if hasattr(self, 'parent') and self.parent is not None:
            return self.parent
        else:
            # This might be the root tracker, 
            # but it might also just be a tracker directly 
            # instantiated from a directory. We look at the parent 
            # directories (remember this tracker is likely in a 
            # subfolder of the actual parents "children" directory) 
            # and try to see whether it's a tracker directory.
            tentative_parent_dir = self.output_dir.parent.parent
            tracker_path = tentative_parent_dir / 'tracker_info.json'
            if tracker_path.exists():
                parent = ExperimentTracker(tentative_parent_dir)
                return parent
        
        return None

    def get_progenitor(self):
        """Return the root level ExperimentTracker"""
        node = self
        parent = node.get_parent()
        while parent is not None:
            node = parent
            parent = node.get_parent()
        return node


def find_experiments(experiment_root: Path):
    # All experiments has a file called tracker info
    experiments_paths = set([p.parent for p in experiment_root.glob('**/tracker_info.json')])
    return experiments_paths
    

def find_experiment_top_level_models(path: Path):
    "Finds all top level experiments with logged models in the given path"
    # Perhaps not the best way of ensuring we get the root experiments is to check common prefixes for all folders with
    # a experiment_metadata subdirectory
    experiments_paths = set([p.parent for p in path.glob('**/models')])
    path_prefixes = defaultdict(list)
    for p in sorted(experiments_paths):
        for prefix in itertools.accumulate(p.parts, lambda a, b: Path(a) / b):
            if prefix in experiments_paths:
                path_prefixes[prefix].append(p)
                break

    fixed_experiment_paths = sorted(path_prefixes.keys())
    return fixed_experiment_paths


def find_top_level_resamples(path: Path):
    "Finds all top level resamples in the given path"
    
    experiments_paths = set(path.glob('**/children/*'))
    path_prefixes = defaultdict(list)
    
    # This code looks for the shortest prefix, gradually constructing the whole path, 
    # but aborting when it finds a match in experiment_paths. 
    # This keeps nested resamples from showing up
    for p in experiments_paths:
        # accumulate will gradually build up the path until 
        # the result matches a path in experiment paths. This means that paths 
        # which are children to some path in experiment_path will be filtered out.
        for prefix in itertools.accumulate(p.parts, lambda a, b: Path(a) / b):
            if prefix in experiments_paths:
                # If there is a match for the currenly built prefix in 
                # experiment_paths, we break here. This stops nested 
                # resamples from showing up
                path_prefixes[prefix].append(p)
                break

    fixed_experiment_paths = sorted(path_prefixes.keys())
    return fixed_experiment_paths