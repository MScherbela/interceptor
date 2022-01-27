<template>
  <div class="ship">
    <div class="card" :class="!ship.alive ? 'sunk' : ''">
      <b-card-header class="pt-2 pb-2">
        <b-row>
          <b-col>
            <b-card-title class="mb-0">{{ ship.name }}</b-card-title>
          </b-col>
          <b-col class="col-md-auto">
                            <b-button class="btn-secondary btn-sm" v-on:click="$store.commit('sink_ship', ship.id)">Sink</b-button>

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
            {{ ship.pos.x }} / {{ ship.pos.y }} <br>
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

</style>