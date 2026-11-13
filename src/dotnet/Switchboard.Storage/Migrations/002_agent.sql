-- The receptionist agent: what it said, the messages it took, the callbacks it booked, and call summaries.

ALTER TABLE transcripts ADD COLUMN role text NOT NULL DEFAULT 'caller' CHECK (role IN ('caller', 'agent'));

ALTER TABLE conversations ADD COLUMN summary text;
ALTER TABLE conversations ADD COLUMN outcome text;

CREATE TABLE messages (
  tenant_id text NOT NULL,
  event_id text NOT NULL, -- The event that created it: redelivery can't make a second copy.
  conversation_id text NOT NULL,
  caller_name text NOT NULL,
  callback_number text NOT NULL,
  message text NOT NULL,
  urgent boolean NOT NULL DEFAULT false,
  for_department text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, event_id),
  FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE callbacks (
  tenant_id text NOT NULL,
  event_id text NOT NULL,
  conversation_id text NOT NULL,
  caller_name text NOT NULL,
  callback_number text NOT NULL,
  preferred_time text NOT NULL,
  reason text NOT NULL DEFAULT '',
  status text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, event_id),
  FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations (tenant_id, id) ON DELETE CASCADE
);

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['messages', 'callbacks']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
      t);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO switchboard_app', t);
  END LOOP;
END $$;
