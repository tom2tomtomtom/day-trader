"use client";

export function DashboardSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Automation Control Skeleton */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-zinc-800 rounded-lg" />
            <div>
              <div className="h-5 w-32 bg-zinc-800 rounded mb-2" />
              <div className="h-4 w-48 bg-zinc-800 rounded" />
            </div>
          </div>
        </div>
        <div className="grid grid-cols-4 gap-2 mb-6">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-10 bg-zinc-800 rounded-lg" />
          ))}
        </div>
        <div className="flex gap-2">
          <div className="flex-1 h-10 bg-zinc-800 rounded-lg" />
          <div className="flex-1 h-10 bg-zinc-800 rounded-lg" />
        </div>
      </div>

      {/* Fear & Greed + Top Picks Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <div className="h-5 w-32 bg-zinc-800 rounded mx-auto mb-4" />
          <div className="h-40 bg-zinc-800 rounded-lg" />
        </div>
        <div className="lg:col-span-2 bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <div className="h-5 w-24 bg-zinc-800 rounded mb-4" />
          <div className="grid grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-32 bg-zinc-800 rounded-lg" />
            ))}
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
            <div className="flex justify-between mb-2">
              <div className="h-4 w-24 bg-zinc-800 rounded" />
              <div className="w-5 h-5 bg-zinc-800 rounded" />
            </div>
            <div className="h-8 w-32 bg-zinc-800 rounded mb-1" />
            <div className="h-4 w-20 bg-zinc-800 rounded" />
          </div>
        ))}
      </div>

      {/* Chart and Positions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <div className="h-5 w-24 bg-zinc-800 rounded mb-4" />
          <div className="h-80 bg-zinc-800 rounded-lg" />
        </div>
        <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
          <div className="h-5 w-32 bg-zinc-800 rounded mb-4" />
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-20 bg-zinc-800 rounded-lg" />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export function PositionsSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-8 w-32 bg-zinc-800 rounded" />

      {/* Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-zinc-900 rounded-xl p-4 border border-zinc-800">
            <div className="h-4 w-20 bg-zinc-800 rounded mb-2" />
            <div className="h-8 w-16 bg-zinc-800 rounded" />
          </div>
        ))}
      </div>

      {/* Positions List */}
      <div className="bg-zinc-900 rounded-xl border border-zinc-800">
        <div className="p-4 border-b border-zinc-800">
          <div className="h-5 w-40 bg-zinc-800 rounded" />
        </div>
        <div className="divide-y divide-zinc-800">
          {[...Array(5)].map((_, i) => (
            <div key={i} className="p-4">
              <div className="flex justify-between">
                <div>
                  <div className="h-5 w-20 bg-zinc-800 rounded mb-2" />
                  <div className="h-4 w-32 bg-zinc-800 rounded" />
                </div>
                <div className="text-right">
                  <div className="h-5 w-16 bg-zinc-800 rounded mb-2" />
                  <div className="h-4 w-24 bg-zinc-800 rounded" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export function SignalsSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      {/* Market Context */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {[...Array(3)].map((_, i) => (
          <div key={i} className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
            <div className="h-5 w-32 bg-zinc-800 rounded mx-auto mb-4" />
            <div className="h-40 bg-zinc-800 rounded-lg" />
          </div>
        ))}
      </div>

      {/* Top Picks */}
      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <div className="h-6 w-32 bg-zinc-800 rounded mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-48 bg-zinc-800 rounded-lg" />
          ))}
        </div>
      </div>
    </div>
  );
}

export function WatchlistSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="flex justify-between">
        <div>
          <div className="h-8 w-32 bg-zinc-800 rounded mb-2" />
          <div className="h-4 w-48 bg-zinc-800 rounded" />
        </div>
        <div className="h-10 w-28 bg-zinc-800 rounded-lg" />
      </div>

      <div className="bg-zinc-900 rounded-xl p-6 border border-zinc-800">
        <div className="h-6 w-40 bg-zinc-800 rounded mb-4" />
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-40 bg-zinc-800 rounded-lg" />
          ))}
        </div>
      </div>
    </div>
  );
}
