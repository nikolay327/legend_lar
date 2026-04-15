import os
import json
import glob
import re

from pathlib import Path
import warnings

from string import Formatter


def _custom_formatwarning(msg, category, filename, lineno, line=None):
    """
    Custom formatter for warning messages to shorten file paths.

    Args:
        msg (Warning): The warning message.
        category (type): The warning category.
        filename (str): The path of the file issuing the warning.
        lineno (int): The line number in the file.
        line (str, optional): The source line.

    Returns:
        str: Formatted warning string.
    """
    return f"{os.path.basename(filename)}:{lineno}: {category.__name__}: {msg}\n"

warnings.formatwarning = _custom_formatwarning

class FileDB:
    def __init__(self, working_dir: str, experiment: str):
        self.experiment = experiment
        self.working_dir = working_dir

        file_db_path = f'{working_dir}/meta/file-db.json'
        with open(file_db_path, 'r') as f:
            self.cfg = json.load(f)

        for name in self.cfg.keys():
            for key in self.cfg[name].keys():
                self.cfg[name][key] = self.cfg[name][key].replace("{experiment}", self.experiment).replace("{working_dir}", self.working_dir)

    def _build_pattern(self, tier, overload=True, **kw):
        """
        Fill the tier-specific file_format template using provided kwargs,
        using '*' wildcards for any missing fields.

        If overload: queries not existing in the pattern will be appended as existing_queries/unknown_queries

        Args:
            **kw: Placeholder values (e.g., datatype, period, run, , timestamp).

        Returns:
            str: A glob-compatible pattern string.
        """

        fmt = self.cfg["file_format"][tier]
        parts = []

        known_fields = set()

        for literal_text, field_name, format_spec, _ in Formatter().parse(fmt):
            parts.append(literal_text)

            if field_name is None:
                continue

            known_fields.add(field_name)

            if field_name in kw and kw[field_name] is not None:
                value = kw[field_name]
                if format_spec:
                    parts.append(format(value, format_spec))
                else:
                    parts.append(str(value))
            else:
                if format_spec and format_spec.endswith("d"):
                    m = re.fullmatch(r"0?(\d+)d", format_spec)
                    if m:
                        width = int(m.group(1))
                        parts.append("?" * width)
                    else:
                        parts.append("*")
                else:
                    parts.append("*")

        if overload:
            for k, v in kw.items():
                if k not in known_fields:
                    parts.append(f"/{v if v is not None else '*'}")

        return "".join(parts)
    
    def build_file(self, tier: str, **kwargs):
        base_dir = Path(self.cfg['tier_dirs'][tier])
        pattern = self._build_pattern(tier, overload=False, **kwargs).lstrip('/')
        # join and glob
        full_pattern = str(base_dir / pattern)
        return full_pattern

    def find_files(self, tier: str, **kwargs):
        """
        Find and return sorted list of files matching the pattern for this tier.

        Args:
            **kwargs: Must include at least experiment, datatype, period, run;
                      timestamp may be omitted to wildcard.

        Returns:
            List[str]: Absolute file paths matching the pattern.
        """
        full_pattern = self.build_file(tier, **kwargs)
        return sorted(glob.glob(full_pattern))

    def _build_regex_from_pattern(self, tier: str):
        """
        Convert the tier-specific file_format template into a named-group regex
        for parsing metadata fields from actual file paths.

        Returns:
            str: A regex pattern with named groups matching the placeholders.
        """
        pattern = self.cfg['file_format'][tier]

        if pattern.startswith("/"):
            pattern = pattern[1:]

        seen = set()
        int_fields = set()
        regex = ""

        for literal_text, field_name, format_spec, _ in Formatter().parse(pattern):
            regex += re.escape(literal_text)

            if field_name is None:
                continue

            if field_name not in seen:
                if format_spec and format_spec.endswith("d"):
                    m = re.fullmatch(r"0?(\d+)d", format_spec)
                    if m:
                        width = int(m.group(1))
                        regex += rf"(?P<{field_name}>\d{{{width}}})"
                    else:
                        regex += rf"(?P<{field_name}>\d+)"
                    int_fields.add(field_name)
                else:
                    regex += rf"(?P<{field_name}>[^/]+)"
                seen.add(field_name)
            else:
                regex += rf"(?P={field_name})"

        return regex, int_fields

    def decode_metadata_from_path(self, tier: str, path_to_file: str):
        """
        Extract metadata fields (period, datatype, run, etc.) from a file path.

        Args:
            path_to_file (str): File path to parse.

        Raises:
            ValueError: If the path doesn't match the expected format.

        Returns:
            dict: Mapping of placeholder names to their string values.
        """
        norm = path_to_file.replace(os.sep, "/")
        regex, int_fields = self._build_regex_from_pattern(tier)
        m = re.match(r"(?:^|.*/)" + regex + r"$", norm)

        if m is None:
            raise ValueError(f"Path does not match expected pattern: {path_to_file}")

        out = m.groupdict()
        for key in int_fields:
            out[key] = int(out[key])

        return out
    
    def parse_metadata_to_path(self, target_tier: str, **metadata):
        return self.cfg["tier_dirs"][target_tier] + self._build_pattern(tier=target_tier, overload=False, **metadata)
