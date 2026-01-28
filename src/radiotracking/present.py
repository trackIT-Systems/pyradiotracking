import argparse
import collections
import datetime
import hashlib
import logging
import os
import threading
from ast import literal_eval
from typing import DefaultDict, Deque, Dict, Iterable, List, Optional, Tuple, Union

import dash
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output, State
from werkzeug.serving import ThreadedWSGIServer

from radiotracking import AbstractMessage, MatchingSignal, Signal
from radiotracking.__main__ import Runner
from radiotracking.consume import AbstractConsumer

SDR_COLORS: DefaultDict[Union[str, int], str] = collections.defaultdict(lambda: "grey")
SDR_COLORS.update(
    {
        "0": "blue",
        "1": "orange",
        "2": "red",
        "3": "green",
        "green": "green",
        "red": "red",
        "yellow": "yellow",
        "blue": "blue",
    }
)


def group(sigs: Iterable[Signal], by: str) -> List[Tuple[str, List[Signal]]]:
    """Optimized grouping using defaultdict for O(n) instead of O(n*m)."""
    grouped: DefaultDict[str, List[Signal]] = collections.defaultdict(list)
    for sig in sigs:
        grouped[sig.__dict__[by]].append(sig)
    return sorted(grouped.items())


class Dashboard(AbstractConsumer, threading.Thread):
    def __init__(
        self,
        device: List[str],
        calibrate: bool,
        calibration: List[float],
        dashboard_host: str,
        dashboard_port: int,
        dashboard_signals: int,
        signal_min_duration_ms: int,
        signal_max_duration_ms: int,
        signal_threshold_dbw: float,
        snr_threshold_db: float,
        sample_rate: int,
        center_freq: int,
        signal_threshold_dbw_max: float = -20,
        snr_threshold_db_max: float = 50,
        **kwargs,
    ):
        threading.Thread.__init__(self)
        self.device = device
        self.calibrate = calibrate
        self.calibration = calibration
        self.signal_queue: Deque[Signal] = collections.deque(maxlen=dashboard_signals)
        self.matched_queue: Deque[MatchingSignal] = collections.deque(maxlen=dashboard_signals)

        # compute boundaries for sliders and initialize filters
        frequency_min = center_freq - sample_rate / 2
        frequency_max = center_freq + sample_rate / 2

        self.app = dash.Dash(
            __name__,
            url_base_pathname="/radiotracking/",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        )

        graph_columns = html.Div(children=[], style={"columns": "2 359px"})
        graph_columns.children.append(dcc.Graph(id="signal-noise", style={"break-inside": "avoid-column"}))
        
        # Shared callback to filter and store signals
        # Use State for sliders to reduce callback frequency (only update on interval or explicit slider release)
        self.app.callback(
            [Output("filtered-signals-store", "data"), Output("filter-state-store", "data")],
            [
                Input("update", "n_intervals"),
            ],
            [
                State("power-slider", "value"),
                State("snr-slider", "value"),
                State("frequency-slider", "value"),
                State("duration-slider", "value"),
            ],
        )(self.update_filtered_signals_store)
        
        self.app.callback(
            Output("signal-noise", "figure"),
            [
                Input("filtered-signals-store", "data"),
                Input("power-slider", "value"),
            ],
        )(self.update_signal_noise)

        graph_columns.children.append(dcc.Graph(id="frequency-histogram", style={"break-inside": "avoid-column"}))
        self.app.callback(
            Output("frequency-histogram", "figure"),
            [
                Input("filtered-signals-store", "data"),
                Input("power-slider", "value"),
                Input("frequency-slider", "value"),
            ],
        )(self.update_frequency_histogram)

        graph_columns.children.append(dcc.Graph(id="signal-match", style={"break-inside": "avoid-column"}))
        self.app.callback(
            Output("signal-match", "figure"),
            [
                Input("update", "n_intervals"),
            ],
        )(self.update_signal_match)

        graph_columns.children.append(dcc.Graph(id="signal-variance", style={"break-inside": "avoid-column"}))
        self.app.callback(
            Output("signal-variance", "figure"),
            [
                Input("filtered-signals-store", "data"),
                Input("power-slider", "value"),
            ],
        )(self.update_signal_variance)

        graph_tab = dcc.Tab(label="tRackIT Signals", children=[])
        graph_tab.children.append(
            html.H4(
                "Running in calibration mode.",
                hidden=not calibrate,
                id="calibration-banner",
                style={
                    "text-align": "center",
                    "width": "100%",
                    "background-color": "#ffcccb",
                    "padding": "20px",
                },
            )
        )
        self.app.callback(
            Output("calibration-banner", "hidden"),
            [
                Input("update", "n_intervals"),
            ],
        )(self.update_calibration_banner)

        graph_tab.children.append(dcc.Interval(id="update", interval=1000))
        self.app.callback(Output("update", "interval"), [Input("interval-slider", "value")])(self.update_interval)
        
        # Shared data store for filtered signals to avoid redundant filtering
        graph_tab.children.append(dcc.Store(id="filtered-signals-store", data=None))
        graph_tab.children.append(dcc.Store(id="filter-state-store", data=None))

        graph_tab.children.append(
            html.Div(
                [
                    dcc.Graph(id="signal-time"),
                ]
            )
        )
        self.app.callback(
            Output("signal-time", "figure"),
            [
                Input("filtered-signals-store", "data"),
                Input("power-slider", "value"),
            ],
        )(self.update_signal_time)

        graph_columns.children.append(html.Div(children=[], id="calibration_output"))
        self.app.callback(
            Output("calibration_output", "children"),
            [
                Input("update", "n_intervals"),
            ],
        )(self.update_calibration)

        graph_columns.children.append(
            html.Div(
                id="settings",
                style={"break-inside": "avoid-column"},
                children=[
                    html.H2("Vizualization Filters"),
                    html.H3("Signal Power"),
                    dcc.RangeSlider(
                        id="power-slider",
                        min=signal_threshold_dbw,
                        max=signal_threshold_dbw_max,
                        step=0.1,
                        value=[signal_threshold_dbw, signal_threshold_dbw_max],
                        marks={
                            int(signal_threshold_dbw): f"{signal_threshold_dbw} dBW",
                            int(signal_threshold_dbw_max): f"{signal_threshold_dbw_max} dBW",
                        },
                    ),
                    html.H3("SNR"),
                    dcc.RangeSlider(
                        id="snr-slider",
                        min=snr_threshold_db,
                        max=snr_threshold_db_max,
                        step=0.1,
                        value=[snr_threshold_db, snr_threshold_db_max],
                        marks={
                            int(snr_threshold_db): f"{snr_threshold_db} dBW",
                            int(snr_threshold_db_max): f"{snr_threshold_db_max} dBW",
                        },
                    ),
                    html.H3("Frequency Range"),
                    dcc.RangeSlider(
                        id="frequency-slider",
                        min=frequency_min,
                        max=frequency_max,
                        step=1,
                        marks={
                            int(frequency_min): f"{frequency_min/1000/1000:.2f} MHz",
                            int(center_freq): f"{center_freq/1000/1000:.2f} MHz",
                            int(frequency_max): f"{frequency_max/1000/1000:.2f} MHz",
                        },
                        value=[frequency_min, frequency_max],
                        allowCross=False,
                    ),
                    html.H3("Signal Duration"),
                    dcc.RangeSlider(
                        id="duration-slider",
                        min=signal_min_duration_ms,
                        max=signal_max_duration_ms,
                        step=0.1,
                        marks={
                            int(signal_min_duration_ms): f"{signal_min_duration_ms} ms",
                            int(signal_max_duration_ms): f"{signal_max_duration_ms} ms",
                        },
                        value=[signal_min_duration_ms, signal_max_duration_ms],
                        allowCross=False,
                    ),
                    html.H2("Dashboard Update Interval"),
                    dcc.Slider(
                        id="interval-slider",
                        min=0.1,
                        max=10,
                        step=0.1,
                        value=1.0,
                        marks={
                            0.1: "0.1 s",
                            1: "1 s",
                            5: "5 s",
                            10: "10 s",
                        },
                    ),
                ],
            )
        )
        graph_tab.children.append(graph_columns)

        tabs = dcc.Tabs(children=[])
        tabs.children.append(graph_tab)

        self.app.layout = html.Div([tabs])
        self.app.layout.style = {"font-family": "sans-serif"}
        self.app.logger.setLevel(logging.WARNING)

        self.server = ThreadedWSGIServer(dashboard_host, dashboard_port, self.app.server)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)

        self.calibrations: Dict[float, Dict[str, float]] = {}
        
        # Performance optimization: filter caching
        self._filter_cache: Optional[Tuple[tuple, List[Signal]]] = None
        self._last_filter_state: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
        self._cache_invalidation_counter = 0
        
        # Precomputed signal data cache (duration_ms)
        # Limit cache size to 2x max signal queue to prevent unbounded growth
        self._signal_data_cache: Dict[int, float] = {}
        self._max_cache_size = dashboard_signals * 2
        
        # Calibration dict size limit
        self._max_calibration_entries = 1000

    def add(self, signal: AbstractMessage):
        if isinstance(signal, Signal):
            self.signal_queue.append(signal)
            
            # Invalidate filter cache when new signal is added
            self._cache_invalidation_counter += 1
            self._filter_cache = None

            # create / update calibration dict calibrations[freq][device] = max(sig.avg)
            if signal.frequency not in self.calibrations:
                self.calibrations[signal.frequency] = {}

            # if freq has no avg for device, set it, else update with max of old and new
            if signal.device not in self.calibrations[signal.frequency]:
                self.calibrations[signal.frequency][signal.device] = signal.avg
            else:
                self.calibrations[signal.frequency][signal.device] = max(
                    self.calibrations[signal.frequency][signal.device], signal.avg
                )
            
            # Limit calibration dict size - remove oldest entries if over limit
            if len(self.calibrations) > self._max_calibration_entries:
                # Remove entries with lowest max values
                sorted_items = sorted(
                    self.calibrations.items(),
                    key=lambda item: max(item[1].values()) if item[1] else float('-inf')
                )
                # Keep only the top entries
                self.calibrations = dict(sorted_items[-self._max_calibration_entries:])

        elif isinstance(signal, MatchingSignal):
            self.matched_queue.append(signal)

    def update_calibration(self, n):
        header = html.Tr(
            children=[
                html.Th("Frequency (MHz)"),
                html.Th("Max (dBW)"),
            ],
            style={"text-align": "left"},
        )

        settings_row = html.Tr(
            children=[
                html.Td("[current settings]"),
                html.Td(""),
            ]
        )

        for device, calibration in zip(self.device, self.calibration):
            header.children.append(html.Th(f"SDR {device} (dB)"))
            settings_row.children.append(html.Td(f"{calibration:.2f}"))

        table = html.Table(children=[header, settings_row], style={"width": "100%", "text-align": "left"})

        for freq, avgs in sorted(self.calibrations.items(), key=lambda item: max(item[1])):
            ordered_avgs = [
                avgs[d] + old if d in avgs else float("-inf") for d, old in zip(self.device, self.calibration)
            ]
            freq_max = max(ordered_avgs)

            row = html.Tr(children=[html.Td(f"{freq/1000/1000:.3f}"), html.Td(f"{freq_max:.2f}")])
            for avg in ordered_avgs:
                row.children.append(html.Td(f"{avg - freq_max:.2f}"))

            table.children.append(row)

        return html.Div(
            [
                html.H2("Calibration Table"),
                table,
            ],
            style={"break-inside": "avoid-column"},
        )

    def update_calibration_banner(self, n):
        return not self.calibrate

    def update_interval(self, interval):
        return interval * 1000
    
    def update_filtered_signals_store(self, n_intervals, power, snr, freq, duration):
        """Shared callback to filter signals and store in dcc.Store for other callbacks.
        Uses State for sliders to reduce callback frequency (debouncing effect)."""
        if power is None or snr is None or freq is None or duration is None:
            return None, None
        
        filtered_sigs = self.select_sigs(power, snr, freq, duration)
        
        # Serialize signals for storage (store only essential data)
        serialized = []
        for sig in filtered_sigs:
            serialized.append({
                'device': sig.device,
                'ts': sig.ts.isoformat() if isinstance(sig.ts, datetime.datetime) else str(sig.ts),
                'frequency': sig.frequency,
                'avg': sig.avg,
                'snr': sig.snr,
                'std': sig.std,
                'duration_ms': self._get_duration_ms(sig),
            })
        
        filter_state = {
            'power': power,
            'snr': snr,
            'freq': freq,
            'duration': duration,
        }
        
        return serialized, filter_state

    def _get_duration_ms(self, sig: Signal) -> float:
        """Get duration in milliseconds, using cache if available."""
        sig_id = id(sig)
        if sig_id not in self._signal_data_cache:
            # Clean up cache if it's too large (remove oldest entries)
            if len(self._signal_data_cache) >= self._max_cache_size:
                # Remove oldest 25% of entries (simple FIFO-like cleanup)
                keys_to_remove = list(self._signal_data_cache.keys())[:self._max_cache_size // 4]
                for key in keys_to_remove:
                    del self._signal_data_cache[key]
            self._signal_data_cache[sig_id] = sig.duration.total_seconds() * 1000
        return self._signal_data_cache[sig_id]
    
    def select_sigs(self, power: List[float], snr: List[float], freq: List[float], duration: List[float]):
        """Select signals matching filters, using cache when filters haven't changed."""
        # Create filter state tuple for comparison
        filter_state = (
            power[0], power[1],
            snr[0], snr[1],
            freq[0], freq[1],
            duration[0], duration[1]
        )
        
        # Check if we can use cached results
        if (self._filter_cache is not None and 
            self._last_filter_state == filter_state):
            return self._filter_cache[1]
        
        # Filter signals
        filtered = [
            sig
            for sig in self.signal_queue
            if sig.avg > power[0]
            and sig.avg < power[1]
            and sig.snr > snr[0]
            and sig.snr < snr[1]
            and sig.frequency > freq[0]
            and sig.frequency < freq[1]
            and self._get_duration_ms(sig) > duration[0]
            and self._get_duration_ms(sig) < duration[1]
        ]
        
        # Cache the results
        self._filter_cache = (filter_state, filtered)
        self._last_filter_state = filter_state
        
        return filtered

    def _deserialize_signals(self, serialized_data):
        """Convert serialized signal data back to Signal-like objects for processing."""
        if not serialized_data:
            return []
        
        class SignalProxy:
            def __init__(self, data):
                self.device = data['device']
                self.ts = datetime.datetime.fromisoformat(data['ts']) if isinstance(data['ts'], str) else data['ts']
                self.frequency = data['frequency']
                self.avg = data['avg']
                self.snr = data['snr']
                self.std = data['std']
                self.duration_ms = data['duration_ms']
        
        return [SignalProxy(d) for d in serialized_data]
    
    def update_signal_time(self, serialized_data, power):
        traces = []
        if not serialized_data:
            fig = go.Figure(data=traces)
            fig.update_layout(
                uirevision="signal-time",  # Preserve zoom/pan state for x-axis
                xaxis={"title": "Time"},
                yaxis={
                    "title": "Signal Power (dBW)", 
                    "range": power if power else None,
                    "fixedrange": True,  # Lock y-axis to slider range, prevent zooming
                },
                legend={"title": "SDR Receiver"},
            )
            return fig
        
        sigs = self._deserialize_signals(serialized_data)
        
        # Group by device using optimized defaultdict
        grouped: DefaultDict[str, List] = collections.defaultdict(list)
        for sig in sigs:
            grouped[sig.device].append(sig)

        for trace_sdr, sdr_sigs in sorted(grouped.items()):
            trace = go.Scatter(
                x=[sig.ts for sig in sdr_sigs],
                y=[sig.avg for sig in sdr_sigs],
                name=trace_sdr,
                mode="markers",
                marker=dict(
                    size=[sig.duration_ms for sig in sdr_sigs],
                    opacity=0.5,
                    color=SDR_COLORS[trace_sdr],
                ),
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            uirevision="signal-time",  # Preserve zoom/pan state for x-axis
            xaxis={"title": "Time", "range": (sigs[0].ts if sigs else None, datetime.datetime.now())},
            yaxis={
                "title": "Signal Power (dBW)", 
                "range": power,
                "fixedrange": True,  # Lock y-axis to slider range, prevent zooming
            },
            legend={"title": "SDR Receiver"},
        )
        return fig

    def update_signal_noise(self, serialized_data, power):
        traces = []
        if not serialized_data:
            fig = go.Figure(data=traces)
            fig.update_layout(
                uirevision="signal-noise",  # Preserve zoom/pan state
                title="Signal to Noise",
                xaxis={"title": "SNR (dB)"},
                yaxis={"title": "Signal Power (dBW)", "range": power if power else None},
                legend={"title": "SDR Receiver"},
            )
            return fig
        
        sigs = self._deserialize_signals(serialized_data)
        
        # Group by device using optimized defaultdict
        grouped: DefaultDict[str, List] = collections.defaultdict(list)
        for sig in sigs:
            grouped[sig.device].append(sig)

        for trace_sdr, sdr_sigs in sorted(grouped.items()):
            trace = go.Scatter(
                x=[sig.snr for sig in sdr_sigs],
                y=[sig.avg for sig in sdr_sigs],
                name=trace_sdr,
                mode="markers",
                marker=dict(
                    size=[sig.duration_ms for sig in sdr_sigs],
                    opacity=0.3,
                    color=SDR_COLORS[trace_sdr],
                ),
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            uirevision="signal-noise",  # Preserve zoom/pan state
            title="Signal to Noise",
            xaxis={"title": "SNR (dB)"},
            yaxis={"title": "Signal Power (dBW)", "range": power},
            legend={"title": "SDR Receiver"},
        )
        return fig

    def update_signal_variance(self, serialized_data, power):
        traces = []
        if not serialized_data:
            fig = go.Figure(data=traces)
            fig.update_layout(
                uirevision="signal-variance",  # Preserve zoom/pan state
                title="Signal Variance",
                xaxis={"title": "Standard Deviation (dB)"},
                yaxis={"title": "Signal Power (dBW)", "range": power if power else None},
                legend={"title": "SDR Receiver"},
            )
            return fig
        
        sigs = self._deserialize_signals(serialized_data)
        
        # Group by device using optimized defaultdict
        grouped: DefaultDict[str, List] = collections.defaultdict(list)
        for sig in sigs:
            grouped[sig.device].append(sig)

        for trace_sdr, sdr_sigs in sorted(grouped.items()):
            trace = go.Scatter(
                x=[sig.std for sig in sdr_sigs],
                y=[sig.avg for sig in sdr_sigs],
                name=trace_sdr,
                mode="markers",
                marker=dict(
                    size=[sig.duration_ms for sig in sdr_sigs],
                    opacity=0.3,
                    color=SDR_COLORS[trace_sdr],
                ),
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            uirevision="signal-variance",  # Preserve zoom/pan state
            title="Signal Variance",
            xaxis={"title": "Standard Deviation (dB)"},
            yaxis={"title": "Signal Power (dBW)", "range": power},
            legend={"title": "SDR Receiver"},
        )
        return fig

    def update_frequency_histogram(self, serialized_data, power, freq):
        traces = []
        if not serialized_data:
            fig = go.Figure(data=traces)
            fig.update_layout(
                uirevision="frequency-histogram",  # Preserve zoom/pan state
                title="Frequency Usage",
                xaxis={"title": "Frequency (MHz)", "range": freq if freq else None},
                yaxis={"title": "Signal Power (dBW)", "range": power if power else None},
                legend_title_text="SDR Receiver",
            )
            return fig
        
        sigs = self._deserialize_signals(serialized_data)
        
        # Group by device using optimized defaultdict
        grouped: DefaultDict[str, List] = collections.defaultdict(list)
        for sig in sigs:
            grouped[sig.device].append(sig)

        for trace_sdr, sdr_sigs in sorted(grouped.items()):
            trace = go.Scatter(
                x=[sig.frequency for sig in sdr_sigs],
                y=[sig.avg for sig in sdr_sigs],
                name=trace_sdr,
                mode="markers",
                marker=dict(
                    size=[sig.duration_ms for sig in sdr_sigs],
                    opacity=0.3,
                    color=SDR_COLORS[trace_sdr],
                ),
            )
            traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            uirevision="frequency-histogram",  # Preserve zoom/pan state
            title="Frequency Usage",
            xaxis={"title": "Frequency (MHz)", "range": freq},
            yaxis={"title": "Signal Power (dBW)", "range": power},
            legend_title_text="SDR Receiver",
        )
        return fig

    def update_signal_match(self, n):
        traces = []

        completed_signals = [msig for msig in self.matched_queue if len(msig._sigs) == 4]

        trace = go.Scatter(
            x=[msig._sigs["0"].avg - msig._sigs["2"].avg for msig in completed_signals],
            y=[msig._sigs["1"].avg - msig._sigs["3"].avg for msig in completed_signals],
            mode="markers",
            marker=dict(
                color=[msig.ts.timestamp() for msig in completed_signals],
                colorscale="Cividis_r",
                opacity=0.5,
            ),
        )
        traces.append(trace)

        fig = go.Figure(data=traces)
        fig.update_layout(
            uirevision="signal-match",  # Preserve zoom/pan state
            title="Matched Frequencies",
            xaxis={
                "title": "Horizontal Difference",
                "range": [-50, 50],
            },
            yaxis={
                "title": "Vertical Difference",
                "range": [-50, 50],
            },
        )
        return fig

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()


class ConfigDashboard(threading.Thread):
    def __init__(
        self,
        running_args: argparse.Namespace,
        immutable_args: Iterable[str],
        dashboard_host: str,
        dashboard_port: int,
        **kwargs,
    ):
        threading.Thread.__init__(self)
        self.app = dash.Dash(
            __name__,
            url_base_pathname="/radiotracking-config/",
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
        )

        config_columns = html.Div(children=[], style={"columns": "2 359px", "padding": "20pt"})
        config_tab = dcc.Tab(label="tRackIT Configuration", children=[config_columns])
        config_columns.children.append(
            html.Div(
                "Reconfiguration requires restarting of radiotracking. Please keep in mind, that a broken configuration might lead to failing starts."
            )
        )

        self.running_args = running_args
        self.immutable_args = immutable_args
        self.config_states: List[State] = []

        for group in Runner.parser._action_groups:
            # skip untitled groups
            if not isinstance(group.title, str):
                continue

            # skip groups not used in the config file
            if len(group._group_actions) == 0:
                continue

            group_div = html.Div(children=[], style={"break-inside": "avoid-column"})
            config_columns.children.append(group_div)

            group_div.children.append(html.H3(f"[{group.title}]"))

            # iterate actions and extract values
            for action in group._group_actions:
                if action.dest not in vars(running_args):
                    continue

                value = vars(running_args)[action.dest]

                group_div.children.append(
                    html.P(
                        children=[
                            html.B(action.dest),
                            f" - {action.help}",
                            html.Br(),
                            dcc.Input(id=action.dest, value=repr(value)),
                        ]
                    )
                )

                if action.type == int or isinstance(action, argparse._CountAction):
                    if not isinstance(value, list):
                        group_div.children[-1].children[-1].type = "number"
                        group_div.children[-1].children[-1].step = 1
                elif action.type == float:
                    if not isinstance(value, list):
                        group_div.children[-1].children[-1].type = "number"
                elif action.type == str:
                    group_div.children[-1].children[-1].type = "text"
                    if isinstance(value, list):
                        group_div.children[-1].children[-1].value = repr(value)
                    else:
                        group_div.children[-1].children[-1].value = value
                elif isinstance(action, argparse._StoreTrueAction):
                    group_div.children[-1].children[-1] = dcc.Checklist(
                        id=action.dest,
                        options=[
                            {"value": action.dest, "disabled": action.dest in self.immutable_args},
                        ],
                        value=[action.dest] if value else [],
                    )

                if action.dest in self.immutable_args:
                    group_div.children[-1].children[-1].disabled = True

                self.config_states.append(State(action.dest, "value"))

        config_columns.children.append(html.Button("Save", id="submit-config"))
        self.app.callback(
            Output("config-msg", "children"),
            [
                Input("submit-config", "n_clicks"),
            ],
            self.config_states,
        )(self.submit_config)

        config_columns.children.append(html.Button("Restart", id="submit-restart"))
        self.app.callback(
            Output("submit-restart", "children"),
            [
                Input("submit-restart", "n_clicks"),
            ],
        )(self.submit_restart)
        config_columns.children.append(html.H4("", id="config-msg", style={"text-align": "center", "padding": "10px"}))

        tabs = dcc.Tabs(children=[])
        tabs.children.append(config_tab)

        self.app.layout = html.Div([tabs])
        self.app.layout.style = {"font-family": "sans-serif"}

        self.server = ThreadedWSGIServer(dashboard_host, dashboard_port + 1, self.app.server)

        self.calibrations: Dict[float, Dict[str, float]] = {}

    def _update_values(self):
        for el in self.app.layout._traverse():
            if getattr(el, "id", None):
                if el.id not in self.running_args:
                    continue

                value = vars(self.running_args)[el.id]
                if isinstance(value, bool):
                    el.value = [el.id] if value else []
                else:
                    el.value = repr(value)

    def submit_config(self, clicks, *form_args):
        msg = html.Div(children=[])
        args = self.running_args

        if not clicks:
            return msg

        for dest, value in zip([state.component_id for state in self.config_states], form_args):
            # find corresponding action
            for action in Runner.parser._actions:
                # ignore immutable args
                if action.dest in self.immutable_args:
                    continue

                if action.dest == dest:
                    # boolean values are returned as lists, check if id is set
                    if isinstance(value, list):
                        args.__dict__[dest] = dest in value
                        continue

                    try:
                        args.__dict__[dest] = literal_eval(value)
                    except (ValueError, SyntaxError):
                        args.__dict__[dest] = value
                    except Exception as e:
                        msg.children.append(html.P(f"Error: value for '{dest}' invalid ({repr(e)})."))
                        return msg

        # write config to actual location
        try:
            Runner.parser.write_config(args, open(self.running_args.config, "w"))
        except Exception as e:
            msg.children.append(html.P(str(e)))
            return msg

        self._update_values()
        msg.children.append(html.P(f"Config successfully written to '{args.config}'."))
        return msg

    def submit_restart(self, clicks):
        if not clicks:
            return "Restart"

        # this is oddly specific and should be generalized
        os.system("systemctl restart radiotracking")

        return "Restarting..."

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
