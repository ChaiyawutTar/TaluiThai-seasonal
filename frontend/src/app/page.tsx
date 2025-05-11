"use client"

import { useState, useEffect } from "react"
import { Search, MapPin, Calendar, Filter } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import LocationSearch from "@/components/location-search"
import RecommendationCard from "@/components/recommendation-card"
import { toast } from "sonner"

interface Recommendation {
  ATT_NAME_TH: string
  suitability_score: number
  ATTR_CATAGORY_TH: string
  composite_score?: number
  distance_km?: number
}

interface Season {
  name: string
  value: string
}

export default function Home() {
  const [seasons, setSeasons] = useState<Season[]>([])
  const [selectedSeason, setSelectedSeason] = useState<string>("")
  const [location, setLocation] = useState<string>("")
  const [coordinates, setCoordinates] = useState<string>("")
  const [maxDistance, setMaxDistance] = useState<number>(50)
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [loadingSeasons, setLoadingSeasons] = useState<boolean>(true)

  useEffect(() => {
    fetchSeasons()
  }, [])

  const fetchSeasons = async () => {
    try {
      setLoadingSeasons(true)
      const response = await fetch("http://localhost:5004/api/seasons")
      const data = await response.json()

      if (data.seasons) {
        const formattedSeasons = data.seasons.map((season: string) => ({
          name: season.charAt(0).toUpperCase() + season.slice(1),
          value: season,
        }))
        setSeasons(formattedSeasons)
        if (formattedSeasons.length > 0) {
          setSelectedSeason(formattedSeasons[0].value)
        }
      }
    } catch (error) {
      console.error("Error fetching seasons:", error)
      toast.error("Failed to fetch seasons. Please check if the backend server is running.")
    } finally {
      setLoadingSeasons(false)
    }
  }

  const handleLocationSelect = (name: string, lat: number, lon: number) => {
    setLocation(name)
    setCoordinates(`${lat},${lon}`)
  }

  const handleSearch = async () => {
    if (!selectedSeason) {
      toast.error("Please select a season")
      return
    }

    try {
      setLoading(true)

      // Define the payload with a proper interface
      interface RecommendPayload {
        query_season_name: string
        top_n: number
        weight_suitability: number
        weight_proximity: number
        user_query_location?: string
        max_distance_km?: number
      }

      const payload: RecommendPayload = {
        query_season_name: selectedSeason,
        top_n: 10,
        weight_suitability: 0.7,
        weight_proximity: 0.3,
      }

      if (coordinates) {
        payload.user_query_location = coordinates
        payload.max_distance_km = maxDistance
      }

      const response = await fetch("http://localhost:5004/api/recommend", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()

      if (data.error) {
        throw new Error(data.error)
      }

      setRecommendations(data.recommendations || [])

      if (data.recommendations?.length === 0) {
        toast.info("No attractions found matching your criteria.")
      } else {
        toast.success(`Found ${data.recommendations.length} attractions!`)
      }
    } catch (error) {
      console.error("Error fetching recommendations:", error)
      toast.error("Failed to fetch recommendations. Please try again.")
      setRecommendations([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-blue-50 to-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-blue-800 mb-2">Thai Attractions Recommender</h1>
          <p className="text-lg text-gray-600">
            Discover the perfect attractions in Thailand based on season and location
          </p>
        </div>

        <Card className="mb-8 border-blue-100 shadow-md">
          <CardHeader>
            <CardTitle className="text-blue-800">Find Attractions</CardTitle>
            <CardDescription>
              Select a season and optionally a location to get personalized recommendations
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <Calendar className="h-4 w-4 text-blue-600" />
                Season
              </label>
              {loadingSeasons ? (
                <Skeleton className="h-10 w-full" />
              ) : (
                <Select value={selectedSeason} onValueChange={setSelectedSeason}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a season" />
                  </SelectTrigger>
                  <SelectContent>
                    {seasons.map((season) => (
                      <SelectItem key={season.value} value={season.value}>
                        {season.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium flex items-center gap-2">
                <MapPin className="h-4 w-4 text-blue-600" />
                Location (Optional)
              </label>
              <LocationSearch onSelect={handleLocationSelect} />
              {location && (
                <p className="text-sm text-gray-500 mt-1">
                  Selected: {location} {coordinates && `(${coordinates})`}
                </p>
              )}
            </div>

            {coordinates && (
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <Filter className="h-4 w-4 text-blue-600" />
                  Maximum Distance (km): {maxDistance}
                </label>
                <Slider
                  value={[maxDistance]}
                  min={5}
                  max={500}
                  step={5}
                  onValueChange={(value) => setMaxDistance(value[0])}
                  className="py-4"
                />
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button
              onClick={handleSearch}
              className="w-full bg-blue-600 hover:bg-blue-700"
              disabled={loading || !selectedSeason}
            >
              {loading ? (
                <>
                  <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                  Searching...
                </>
              ) : (
                <>
                  <Search className="mr-2 h-4 w-4" />
                  Find Attractions
                </>
              )}
            </Button>
          </CardFooter>
        </Card>

        {recommendations.length > 0 && (
          <div className="space-y-4">
            <h2 className="text-2xl font-semibold text-blue-800 flex items-center">
              Recommended Attractions
              <Badge variant="outline" className="ml-2 bg-blue-50">
                {recommendations.length} results
              </Badge>
            </h2>
            <Separator className="my-4" />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {recommendations.map((rec, index) => (
                <RecommendationCard key={index} recommendation={rec} />
              ))}
            </div>
          </div>
        )}

        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
            {[...Array(6)].map((_, i) => (
              <Card key={i} className="overflow-hidden">
                <div className="p-4 space-y-3">
                  <Skeleton className="h-6 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-20 w-full" />
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-1/3" />
                    <Skeleton className="h-4 w-1/3" />
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </main>
  )
}
