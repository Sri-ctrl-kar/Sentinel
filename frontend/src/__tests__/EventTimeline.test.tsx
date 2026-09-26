/** The timeline is the navigation device: every marker must seek the video. */

import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { EventTimeline } from '../components/EventTimeline';
import { buildMarkers } from '../lib/timeline';
import { events as eventsFixture, risk as riskFixture } from '../test/fixtures';

const events = eventsFixture.events;
const reports = riskFixture.timeline;

describe('event timeline', () => {
  it('lists every recorded event and every severity change', () => {
    render(
      <EventTimeline
        events={events}
        reports={reports}
        duration={3}
        currentTime={0}
        onSeek={() => {}}
      />,
    );
    const expected = buildMarkers(events, reports);
    expect(screen.getByText(`${expected.length} markers`)).toBeInTheDocument();
  });

  it('shows the event kinds the pipeline emits', () => {
    render(
      <EventTimeline
        events={events}
        reports={reports}
        duration={3}
        currentTime={0}
        onSeek={() => {}}
      />,
    );
    for (const action of Object.keys(eventsFixture.action_counts)) {
      expect(screen.getAllByText(action.replace(/_/g, ' ')).length).toBeGreaterThan(0);
    }
  });

  it('seeks the video to an event timestamp when its row is clicked', async () => {
    const onSeek = vi.fn();
    const user = userEvent.setup();
    render(
      <EventTimeline
        events={events}
        reports={reports}
        duration={3}
        currentTime={0}
        onSeek={onSeek}
      />,
    );

    const markers = buildMarkers(events, reports);
    const target = markers[3];
    await user.click(screen.getByTestId(`timeline-row-${target.key}`));
    expect(onSeek).toHaveBeenCalledWith(target.timestamp);
  });

  it('seeks from the scrub track too', async () => {
    const onSeek = vi.fn();
    const user = userEvent.setup();
    render(
      <EventTimeline
        events={events}
        reports={reports}
        duration={3}
        currentTime={0}
        onSeek={onSeek}
      />,
    );
    const ticks = screen.getByTestId('timeline-track').querySelectorAll('button');
    await user.click(ticks[0]);
    expect(onSeek).toHaveBeenCalled();
  });

  it('gives every marker an accessible name, not just a colour', () => {
    render(
      <EventTimeline
        events={events.slice(0, 2)}
        reports={[]}
        duration={3}
        currentTime={0}
        onSeek={() => {}}
      />,
    );
    const ticks = screen.getByTestId('timeline-track').querySelectorAll('button');
    for (const tick of ticks) {
      expect(tick.textContent).toMatch(/Seek to/);
    }
  });

  it('marks the row nearest the playhead as active', () => {
    const markers = buildMarkers(events, reports);
    const target = markers[2];
    render(
      <EventTimeline
        events={events}
        reports={reports}
        duration={3}
        currentTime={target.timestamp}
        onSeek={() => {}}
      />,
    );
    expect(screen.getByTestId(`timeline-row-${target.key}`).className).toContain(
      'timeline__row--active',
    );
  });

  it('says so when there are no events at all', () => {
    render(
      <EventTimeline
        events={[]}
        reports={[]}
        duration={0}
        currentTime={0}
        onSeek={() => {}}
      />,
    );
    expect(screen.getByText('No events recorded.')).toBeInTheDocument();
  });
});
