<template>
  <div id="app" data="data">
    <b-container fluid>
      <b-row>
        <b-col cols="9">
          <ShipMap/>
        </b-col>
        <b-col>
          <ShipList/>
        </b-col>
      </b-row>
    </b-container>
  </div>
</template>

<script>
import ShipList from './components/ShipLIst.vue'
import ShipMap from "../ShipMap";

export default {
  name: 'App',
  components: {
    ShipList,
    ShipMap
  },

}
</script>

<style>
/*#app {*/
/*  font-family: Avenir, Helvetica, Arial, sans-serif;*/
/*  -webkit-font-smoothing: antialiased;*/
/*  -moz-osx-font-smoothing: grayscale;*/
/*  text-align: center;*/
/*  color: #2c3e50;*/
/*  margin-top: 60px;*/
/*}*/
</style>
