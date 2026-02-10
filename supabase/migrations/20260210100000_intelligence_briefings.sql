-- Intelligence briefings table for dashboard API
CREATE TABLE IF NOT EXISTS intelligence_briefings (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    briefing_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_intel_briefings_created ON intelligence_briefings (created_at DESC);

ALTER TABLE intelligence_briefings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Service role full access" ON intelligence_briefings FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "Anon read access" ON intelligence_briefings FOR SELECT USING (true);
