/**
 * A migration is a declared sequence, not a conversational repair turn.
 *
 * This keeps the real applyMigrationToMain -> dispatchTool -> runExecutor ->
 * executeAction path. The DDL boundary fails deterministically; if migration
 * dispatch ever stops passing allowReplan:false, executeAction reaches the
 * mocked OpenAI client and attempts its replacement CREATE_TABLE action.
 */

const mockQueryRaw = jest.fn()
const mockExecuteRawUnsafe = jest.fn()
const mockGetOpenAIClient = jest.fn()
const mockSnapshotSchema = jest.fn()
const mockRecordSchemaIntent = jest.fn()

const mockPrisma = {
  $queryRaw: (...args: any[]) => mockQueryRaw(...args),
  $executeRawUnsafe: (...args: any[]) => mockExecuteRawUnsafe(...args),
}

jest.mock('@/lib/db/prisma', () => ({ prisma: mockPrisma }))
jest.mock('@/lib/db', () => ({ prisma: mockPrisma }))

jest.mock('@/lib/ai/execution-timeline', () => ({
  trackAction: async (_projectId: string, _runId: string, _action: unknown, work: () => Promise<unknown>) => work(),
}))

jest.mock('@/lib/versioning/schema-versions', () => ({
  snapshotSchema: (...args: any[]) => mockSnapshotSchema(...args),
}))

jest.mock('@/lib/autonomy/intent-conformance', () => ({
  recordSchemaIntent: (...args: any[]) => mockRecordSchemaIntent(...args),
}))

jest.mock('@/lib/services/databaseProvisioning', () => ({
  getWorkspaceDatabaseNames: () => ({ postgresSchema: 'workspace_project_1' }),
}))

jest.mock('@/lib/branches/protection', () => ({
  GATED_SCHEMA_TOOLS: new Set(['create_table']),
  isProductionProtected: async () => false,
  branchRequired: jest.fn(),
}))

jest.mock('@/lib/ai/openai-service', () => ({
  getOpenAIClient: (...args: any[]) => mockGetOpenAIClient(...args),
  trackCompletionCost: jest.fn(),
}))

// `require` intentionally follows the mock values: Jest hoists mock
// declarations, while a static import would load apply-migration before these
// values are initialized.
const { applyMigrationToMain } = require('@/lib/mcp/apply-migration') as typeof import('@/lib/mcp/apply-migration')

describe('applyMigrationToMain no-replanning boundary', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockQueryRaw.mockResolvedValue([])
    mockSnapshotSchema.mockResolvedValue('version-1')
    mockRecordSchemaIntent.mockResolvedValue(undefined)
    mockExecuteRawUnsafe.mockRejectedValue(new Error('forced DDL failure'))
    // If the retry guard is reached with allowReplan=true, this is the action
    // executeAction would run instead of reporting the original failure.
    mockGetOpenAIClient.mockReturnValue({
      chat: {
        completions: {
          create: jest.fn().mockResolvedValue({
            choices: [{ message: { content: JSON.stringify({
              action: 'CREATE_TABLE',
              params: { tableName: 'replacement_table', columns: [{ name: 'body', type: 'TEXT' }] },
            }) } }],
          }),
        },
      },
    })
  })

  it('runs the original action, then reports its failure without model replanning', async () => {
    const result = await applyMigrationToMain(
      'project-1',
      'user-1',
      'CREATE TABLE posts (title text)',
      { workspaceSchema: 'workspace_project_1', agentSurface: true },
    )

    // The original parsed action reached the real executor and failed at DDL.
    expect(mockExecuteRawUnsafe).toHaveBeenCalledTimes(1)
    expect(mockExecuteRawUnsafe.mock.calls[0][0]).toContain('"posts"')

    // extractAction starts by asking this boundary for a model client. No call
    // proves the retry condition did not enter its model-driven branch.
    expect(mockGetOpenAIClient).not.toHaveBeenCalled()
    expect(mockExecuteRawUnsafe.mock.calls.flat().join('\n')).not.toContain('replacement_table')

    expect(result).toMatchObject({
      ok: false,
      code: 'MIGRATION_FAILED',
      error: 'Migration stopped at: CREATE TABLE posts (title text)',
      detail: expect.stringContaining('forced DDL failure'),
      applied: [],
      remaining: [],
    })
  })
})
