<template>
  <div class="ship">
    <div class="card" :class="{sunk: !ship.alive, warship: ship.is_warship()}">
      <b-card-header class="pt-2 pb-2" >
        <b-row>
          <b-col>
            <b-card-title class="mb-0">{{ ship.name }}</b-card-title>
          </b-col>
          <b-col class="col-md-auto">
            <b-button-close v-on:click="$store.commit('remove_ship', id)"/>
          </b-col>
        </b-row>
      </b-card-header>
      <b-card-body>
        <b-row>
          <b-col>
            <b>{{ ship.ship_type }}</b> <br>
            {{ ship.tons }} BRT
          </b-col>
          <b-col>
            {{ ship.pos.x.toFixed(2) }} / {{ ship.pos.y.toFixed(2) }} <br>
            {{ rel_direction.toFixed(0) }} deg / {{ distance.toFixed(2) }} sm <br>
            {{ ship.pos.heading }} deg, {{ ship.pos.speed }} kn
          </b-col>
        </b-row>
      </b-card-body>
    </div>
  </div>
</template>

<script>

export default {
  name: "ShipCard",
  props: {
    id: Number,
    ship: Object
  },
  computed: {
    distance() {
      return this.$store.state.uboot_pos.get_relative_position(this.ship.pos).distance
    },
    rel_direction() {
      return this.$store.state.uboot_pos.get_relative_position(this.ship.pos).rel_direction
    },
  }
}
</script>

<style scoped>
.ship {
  margin-top: 20px;
  margin-bottom: 20px;
}

.sunk {
  opacity: 0.4;
}

.warship .card-body{
  background-color: #d6969c;
}

.warship .card-header{
  background-color: #b34752
}

</style>