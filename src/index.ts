import { McpAgent } from "agents/mcp";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";

type Env = {
  PERPLEXITY_API_KEY: string;
};

type ConversationTurn = { role: "user" | "assistant"; content: string };

type PerplexityResult = {
  answer: string;
  citations: string[];
  threadId: string;
};

// One Durable Object instance per client session. this.ctx.storage
// is where thread history for perplexity_follow_up lives, scoped to
// this session and persisted across calls.
export class PerplexityMCP extends McpAgent<Env> {
  server = new McpServer({
    name: "perplexity-research",
    version: "1.0.0",
  });

  async init() {
    this.server.tool(
      "perplexity_quick_search",
      "Fast web search with a direct answer and sources, good for a quick fact check, a current event, or a single question. Returns an answer plus a thread_id you can pass to perplexity_follow_up to keep asking about the same topic.",
      {
        query: z.string().describe("The question or topic to search for"),
      },
      async ({ query }) => {
        const result = await this.callPerplexity("sonar", [
          { role: "user", content: query },
        ]);
        return toolResult(result);
      }
    );

    this.server.tool(
      "perplexity_deep_research",
      "Slower, thorough multi source research using Perplexity's deep research model. Use for comprehensive topic research, comparisons, or anything that benefits from Perplexity checking many sources rather than a quick answer. Can take one to a few minutes to return. Returns an answer plus a thread_id for follow ups.",
      {
        query: z.string().describe("The research question or topic"),
      },
      async ({ query }) => {
        const result = await this.callPerplexity("sonar-deep-research", [
          { role: "user", content: query },
        ]);
        return toolResult(result);
      }
    );

    this.server.tool(
      "perplexity_follow_up",
      "Ask a follow up question that continues a previous perplexity_quick_search or perplexity_deep_research call, using the thread_id that call returned. Keeps the prior question and answer as context so the follow up doesn't need to repeat it.",
      {
        thread_id: z
          .string()
          .describe("The thread_id returned by an earlier perplexity call"),
        query: z.string().describe("The follow up question"),
      },
      async ({ thread_id, query }) => {
        const history = (await this.getThread(thread_id)) ?? [];
        if (history.length === 0) {
          throw new Error(
            `No thread found for thread_id "${thread_id}". It may have expired or never existed. Start a new perplexity_quick_search or perplexity_deep_research instead.`
          );
        }
        const messages: ConversationTurn[] = [
          ...history,
          { role: "user", content: query },
        ];
        const result = await this.callPerplexity(
          "sonar-pro",
          messages,
          thread_id
        );
        return toolResult(result);
      }
    );
  }

  private async getThread(
    threadId: string
  ): Promise<ConversationTurn[] | undefined> {
    return (
      (await this.ctx.storage.get<ConversationTurn[]>(`thread:${threadId}`)) ??
      undefined
    );
  }

  private async saveThread(threadId: string, turns: ConversationTurn[]) {
    // Trim to the last 20 turns so a long running thread doesn't grow
    // storage or the outgoing request payload without bound.
    const trimmed = turns.slice(-20);
    await this.ctx.storage.put(`thread:${threadId}`, trimmed);
  }

  private async callPerplexity(
    model: string,
    messages: ConversationTurn[],
    existingThreadId?: string
  ): Promise<PerplexityResult> {
    const res = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.env.PERPLEXITY_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, messages }),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`Perplexity API error ${res.status}: ${errText}`);
    }

    const data: any = await res.json();
    const answer: string = data.choices?.[0]?.message?.content ?? "";
    const citations: string[] = data.citations ?? [];

    const threadId = existingThreadId ?? crypto.randomUUID();
    await this.saveThread(threadId, [
      ...messages,
      { role: "assistant", content: answer },
    ]);

    return { answer, citations, threadId };
  }
}

function toolResult(result: PerplexityResult) {
  const sourcesBlock =
    result.citations.length > 0
      ? `\n\nsources:\n${result.citations
          .map((url, i) => `[${i + 1}] ${url}`)
          .join("\n")}`
      : "";

  return {
    content: [
      {
        type: "text" as const,
        text: `${result.answer}${sourcesBlock}\n\nthread_id: ${result.threadId} (pass this to perplexity_follow_up to continue this thread)`,
      },
    ],
  };
}

export default {
  fetch(request: Request, env: Env, ctx: ExecutionContext) {
    const url = new URL(request.url);

    if (url.pathname === "/mcp") {
      return PerplexityMCP.serve("/mcp").fetch(request, env, ctx);
    }

    // SSE kept for older MCP clients that don't support streamable HTTP yet.
    if (url.pathname === "/sse") {
      return PerplexityMCP.serveSSE("/sse").fetch(request, env, ctx);
    }

    return new Response(
      "perplexity mcp server is running. connect a client at /mcp",
      { status: 200 }
    );
  },
};
