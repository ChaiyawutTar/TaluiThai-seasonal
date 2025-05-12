// components/recommendation-card.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { MapPin, Star } from "lucide-react"
import StarRating from "./star-rating"

interface RecommendationProps {
  recommendation: {
    ATT_NAME_TH: string
    suitability_score: number
    ATTR_CATAGORY_TH: string
    composite_score?: number
    distance_km?: number
    ATT_LATITUDE: number
    ATT_LONGITUDE: number
  }
}

export default function RecommendationCard({ recommendation }: RecommendationProps) {
  const { 
    ATT_NAME_TH, 
    suitability_score, 
    ATTR_CATAGORY_TH, 
    composite_score, 
    distance_km,
    ATT_LATITUDE,
    ATT_LONGITUDE
  } = recommendation

  // Format scores as percentages
  const formatScore = (score: number) => {
    return `${Math.round(score * 100)}%`
  }

  // Format distance
  const formatDistance = (distance: number) => {
    if (distance < 1) {
      return `${Math.round(distance * 1000)} m`
    }
    return `${distance.toFixed(1)} km`
  }

  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow border-blue-100">
      <CardHeader className="bg-gradient-to-r from-blue-50 to-blue-100 pb-2">
        <div className="flex justify-between items-start">
          <CardTitle className="text-lg font-bold text-blue-800 line-clamp-2">{ATT_NAME_TH}</CardTitle>
        </div>
        <Badge variant="outline" className="bg-white text-blue-700 mt-1">
          {ATTR_CATAGORY_TH}
        </Badge>
      </CardHeader>
      <CardContent className="pt-4">
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <Star className="h-4 w-4 text-yellow-500 mr-1" />
              <span className="text-sm font-medium">Suitability Score: {formatScore(suitability_score)}</span>
            </div>

            {composite_score && (
              <div className="text-sm font-medium text-blue-600">Match: {formatScore(composite_score)}</div>
            )}
          </div>

          {distance_km !== undefined && (
            <div className="flex items-center text-gray-600">
              <MapPin className="h-4 w-4 mr-1" />
              <span className="text-sm">Distance: {formatDistance(distance_km)}</span>
            </div>
          )}
          
          <div className="border-t border-gray-100 pt-3">
            <StarRating 
              attractionName={ATT_NAME_TH}
              latitude={ATT_LATITUDE}
              longitude={ATT_LONGITUDE}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}