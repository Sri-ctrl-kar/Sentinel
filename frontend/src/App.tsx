/**
 * The Sentinel command center.
 *
 * Three questions, in the order an operator asks them, left to right and top to
 * bottom: what is happening (video + tracks), what will happen and when (risk,
 * prediction), and why Sentinel believes it (evidence, lifecycle, provenance).
 *
 * Everything rendered here came from the API. Where a field is absent the
 * interface says so; it never fills a gap with a plausible value.
 */

import { useCallback, useEffect, useMemo, useState } from 'react';

import { API_BASE_URL, getDevice, getHealth, videoUrl } from './api/client';
import type { DeviceInfo } from './api/types';
import { AnalysisProgress } from './components/AnalysisProgress';
import { ErrorState } from './components/ErrorState';
import { EvidencePanel } from './components/EvidencePanel';
import { EventTimeline } from './components/EventTimeline';
import { Header } from './components/Header';
import { LifecycleTrail } from './components/LifecycleTrail';
import { NarrativePanel } from './components/NarrativePanel';
import { PredictionPanel } from './components/PredictionPanel';
import { RiskPanel } from './components/RiskPanel';
import { TrustPanel } from './components/TrustPanel';
import { UploadLanding } from './components/UploadLanding';
import { VideoIntelligence } from './components/VideoIntelligence';
import { useAnalysis } from './hooks/useAnalysis';
import { clipDuration, peakRisk, riskAt } from './lib/timeline';

export default function App() {
  const analysis = useAnalysis();
  const [connected, setConnected] = useState<boolean | null>(null);
  const [device, setDevice] = useState<DeviceInfo | null>(null);
  const [filename, setFilename] = useState<string | null>(null);
  const [currentTime, setCurrentTime] = useState(0);
  const [seekTo, setSeekTo] = useState<number | null>(null);

  // One probe at startup: is the service there, and what is it running on?
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        await getHealth();
        const info = await getDevice();
        if (cancelled) return;
        setConnected(true);
        setDevice(info.selected);
      } catch {
        if (!cancelled) setConnected(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleSelect = useCallback(
    (file: File) => {
      setFilename(file.name);
      setCurrentTime(0);
      setSeekTo(null);
      void analysis.start(file);
    },
    [analysis],
  );

  const handleReset = useCallback(() => {
    setFilename(null);
    setCurrentTime(0);
    setSeekTo(null);
    analysis.reset();
  }, [analysis]);

  const handleSeek = useCallback((time: number) => {
    setSeekTo(time);
    setCurrentTime(time);
  }, []);

  const reports = analysis.risk?.timeline ?? [];

  // Land on the incident. When the clip contains a worst moment, the playhead
  // starts there rather than at a black first frame, so the answer to "what
  // happened?" is on screen without the operator hunting for it.
  useEffect(() => {
    if (analysis.phase !== 'complete') return;
    const peak = peakRisk(analysis.risk?.timeline ?? []);
    if (peak) {
      setSeekTo(peak.timestamp);
      setCurrentTime(peak.timestamp);
    }
  }, [analysis.phase, analysis.risk]);

  const currentReport = useMemo(
    () => riskAt(reports, currentTime),
    [reports, currentTime],
  );
  const involved = currentReport?.assessments[0]?.involved_entity_ids ?? [];
  const duration = clipDuration(
    analysis.status?.video?.duration_seconds ?? null,
    analysis.timeline?.frames ?? [],
  );

  const deviceForHeader = analysis.status?.device ?? device;
  const busy =
    analysis.phase === 'uploading' ||
    analysis.phase === 'queued' ||
    analysis.phase === 'running' ||
    analysis.phase === 'loading';

  return (
    <div className="app">
      <Header
        connected={connected}
        device={deviceForHeader}
        apiBase={API_BASE_URL}
        onReset={handleReset}
        canReset={analysis.phase === 'complete' || analysis.phase === 'failed'}
      />

      <main className="app__main">
        {connected === false && analysis.phase === 'idle' && (
          <ErrorState
            title="API unavailable"
            message={`Cannot reach the Sentinel API at ${API_BASE_URL}.`}
            hint="Start it with: uvicorn app.api.server:app --reload --port 8000"
          />
        )}

        {analysis.phase === 'idle' && connected !== false && (
          <UploadLanding onSelect={handleSelect} />
        )}

        {busy && (
          <AnalysisProgress
            phase={analysis.phase}
            filename={filename}
            status={analysis.status}
          />
        )}

        {analysis.phase === 'failed' && (
          <ErrorState
            message={analysis.error ?? 'The analysis failed.'}
            hint={
              analysis.status?.error
                ? 'The pipeline could not process this file. A different clip may work.'
                : undefined
            }
            onRetry={handleReset}
          />
        )}

        {analysis.phase === 'complete' && analysis.analysisId && (
          <div className="workspace">
            <div className="workspace__column">
              <VideoIntelligence
                videoSrc={videoUrl(analysis.analysisId)}
                video={analysis.status?.video ?? null}
                frames={analysis.timeline?.frames ?? []}
                currentTime={currentTime}
                onTimeChange={setCurrentTime}
                seekTo={seekTo}
                involved={involved}
              />
              <EventTimeline
                events={analysis.events?.events ?? []}
                reports={reports}
                duration={duration}
                currentTime={currentTime}
                onSeek={handleSeek}
              />
              <NarrativePanel incident={analysis.incident} />
            </div>

            <div className="workspace__column">
              <RiskPanel
                risk={analysis.risk}
                current={currentReport}
                currentTime={currentTime}
              />
              <PredictionPanel incident={analysis.incident} />
              <EvidencePanel incident={analysis.incident} />
              <LifecycleTrail state={analysis.incident?.lifecycle_state ?? null} />
              <TrustPanel
                device={deviceForHeader}
                pipeline={analysis.status?.pipeline ?? null}
                grounding={analysis.incident?.grounding ?? null}
                coordinateSpace={
                  analysis.incident?.evidence?.coordinate_space ??
                  analysis.risk?.coordinate_space ??
                  null
                }
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
