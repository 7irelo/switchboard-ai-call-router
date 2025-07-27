-- Applied by a privileged role. The app role is neither owner nor BYPASSRLS.
-- Every tenant table: tenant_id, ENABLE + FORCE RLS, one policy. tests/dotnet/Isolation audits this.

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'switchboard_app') THEN
    CREATE ROLE switchboard_app LOGIN NOBYPASSRLS NOSUPERUSER PASSWORD 'placeholder-rotated-by-migrator';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'switchboard_resolver') THEN
    CREATE ROLE switchboard_resolver NOLOGIN BYPASSRLS NOSUPERUSER;
  END IF;
END $$;

CREATE TABLE tenants (
  tenant_id text PRIMARY KEY,
  name text NOT NULL,
  current_version int,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE tenant_settings (
  tenant_id text PRIMARY KEY REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  timezone text NOT NULL,
  prompt_override text NOT NULL DEFAULT '',
  document jsonb NOT NULL -- holidays, languages, custom intents, fallback, cost overrides, channels
);

CREATE TABLE schedules (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  id text NOT NULL,
  windows jsonb NOT NULL,
  PRIMARY KEY (tenant_id, id)
);

CREATE TABLE departments (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  id text NOT NULL,
  target text NOT NULL,
  hours text NOT NULL,
  closed_action text NOT NULL DEFAULT 'voicemail' CHECK (closed_action IN ('voicemail', 'reception')),
  emergency_capable boolean NOT NULL DEFAULT false,
  PRIMARY KEY (tenant_id, id)
);

-- Composite foreign keys: a row can't reference another tenant's department.
CREATE TABLE intent_map (
  tenant_id text NOT NULL,
  intent text NOT NULL,
  department_id text NOT NULL,
  PRIMARY KEY (tenant_id, intent),
  FOREIGN KEY (tenant_id, department_id) REFERENCES departments (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE routing_rules (
  tenant_id text NOT NULL,
  id text NOT NULL,
  did text NOT NULL,
  department_id text NOT NULL,
  PRIMARY KEY (tenant_id, id),
  FOREIGN KEY (tenant_id, department_id) REFERENCES departments (tenant_id, id) ON DELETE CASCADE
);

-- Immutable published snapshots. Each decision records the version it used.
CREATE TABLE config_versions (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  version int NOT NULL,
  config jsonb NOT NULL,
  published_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, version)
);

CREATE TABLE channel_bindings (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  kind text NOT NULL CHECK (kind IN ('did', 'whatsapp')),
  value text NOT NULL,
  PRIMARY KEY (kind, value)
);

CREATE TABLE api_credentials (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  key_hash text PRIMARY KEY
);

CREATE TABLE conversations (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  id text NOT NULL,
  channel text NOT NULL,
  config_version text NOT NULL DEFAULT '',
  started_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, id)
);

CREATE TABLE conversation_events (
  tenant_id text NOT NULL,
  event_id text NOT NULL,
  conversation_id text NOT NULL,
  sequence bigint NOT NULL,
  type text NOT NULL,
  payload jsonb NOT NULL,
  occurred_at timestamptz NOT NULL,
  PRIMARY KEY (tenant_id, event_id),
  FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE transcripts (
  tenant_id text NOT NULL,
  conversation_id text NOT NULL,
  turn int NOT NULL,
  text text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, conversation_id, turn),
  FOREIGN KEY (tenant_id, conversation_id) REFERENCES conversations (tenant_id, id) ON DELETE CASCADE
);

CREATE TABLE message_dedup (
  tenant_id text NOT NULL REFERENCES tenants (tenant_id) ON DELETE CASCADE,
  provider_message_id text NOT NULL,
  received_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, provider_message_id)
);

DO $$
DECLARE t text;
BEGIN
  FOREACH t IN ARRAY ARRAY['tenants', 'tenant_settings', 'schedules', 'departments', 'intent_map', 'routing_rules',
                           'config_versions', 'channel_bindings', 'api_credentials', 'conversations',
                           'conversation_events', 'transcripts', 'message_dedup']
  LOOP
    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
    EXECUTE format(
      'CREATE POLICY tenant_isolation ON %I USING (tenant_id = current_setting(''app.tenant_id'', true)) WITH CHECK (tenant_id = current_setting(''app.tenant_id'', true))',
      t);
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON %I TO switchboard_app', t);
  END LOOP;
END $$;

-- The only ways to learn a tenant without already being one. Each returns the bare minimum.
GRANT SELECT ON channel_bindings, api_credentials, tenants TO switchboard_resolver;
GRANT USAGE ON SCHEMA public TO switchboard_resolver;

CREATE FUNCTION resolve_channel(p_kind text, p_value text) RETURNS text
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp
  AS $$ SELECT tenant_id FROM channel_bindings WHERE kind = p_kind AND value = p_value $$;

CREATE FUNCTION authenticate_key(p_hash text) RETURNS text
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp
  AS $$ SELECT tenant_id FROM api_credentials WHERE key_hash = p_hash $$;

-- For background jobs that go through each tenant in turn, each inside its own scope.
CREATE FUNCTION list_tenant_ids() RETURNS SETOF text
  LANGUAGE sql STABLE SECURITY DEFINER SET search_path = public, pg_temp
  AS $$ SELECT tenant_id FROM tenants ORDER BY tenant_id $$;

ALTER FUNCTION resolve_channel(text, text) OWNER TO switchboard_resolver;
ALTER FUNCTION authenticate_key(text) OWNER TO switchboard_resolver;
ALTER FUNCTION list_tenant_ids() OWNER TO switchboard_resolver;
REVOKE ALL ON FUNCTION resolve_channel(text, text), authenticate_key(text), list_tenant_ids() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION resolve_channel(text, text), authenticate_key(text), list_tenant_ids() TO switchboard_app;
